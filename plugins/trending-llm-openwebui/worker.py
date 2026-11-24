#!/usr/bin/env python3
"""Trending Open-Source LLMs + Open WebUI worker for Colab Playground.

Supports one-click launch of top 2026 Hugging Face GGUF models (Qwen3.8-27B,
Gemma 4 26B-A4B / 31B Multimodal Vision, DeepSeek-V4.1-Flash, Qwen3-Coder-Next,
GLM-4.7-Flash, or any custom HF GGUF repo) with automatic VRAM-aware quantisation
selection, automatic mmproj loading, latest prebuilt llama.cpp CUDA binary, and
password-protected Open WebUI 0.12.3.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

E = os.environ.get
RUN_DIR = Path(E("PLUGIN_RUN_DIR", "/root/.colab-playground/runs/manual"))
STATUS_FILE = Path(E("PLUGIN_STATUS_FILE", str(RUN_DIR / "status.json")))
HEARTBEAT = E("PLUGIN_HEARTBEAT", "")

PRESET = (E("PLUGIN_INPUT_PRESET", "Qwen3.8-27B") or "Qwen3.8-27B").strip()
CUSTOM_REPO = (E("PLUGIN_INPUT_CUSTOM_REPO", "") or "").strip()
QUANT_PREF = (E("PLUGIN_INPUT_QUANT", "auto") or "auto").strip()
CTX_SIZE = int(float(E("PLUGIN_INPUT_CTX_SIZE", "16384") or 16384))
ADMIN_EMAIL = (E("PLUGIN_INPUT_ADMIN_EMAIL", "rick@example.com") or "rick@example.com").strip()
ADMIN_PASSWORD = (E("PLUGIN_INPUT_ADMIN_PASSWORD", "") or "").strip()
OW_PORT = int(float(E("PLUGIN_INPUT_WEBUI_PORT", "8081") or 8081))
LLAMA_PORT = int(float(E("PLUGIN_INPUT_API_PORT", "8000") or 8000))
HF_TOKEN = (E("PLUGIN_HF_TOKEN", "") or "").strip()

PRESET_REPOS: dict[str, str] = {
    "Qwen3.8-27B": "unsloth/Qwen3.8-27B-GGUF",
    "Gemma-4-26B-A4B-Multimodal": "unsloth/gemma-4-26B-A4B-it-GGUF",
    "Gemma-4-31B-Multimodal": "unsloth/gemma-4-31B-it-GGUF",
    "DeepSeek-V4.1-Flash": "unsloth/DeepSeek-V4.1-Flash-1017-GGUF",
    "Qwen3-Coder-Next": "unsloth/Qwen3-Coder-Next-GGUF",
    "GLM-4.7-Flash": "unsloth/GLM-4.7-Flash-GGUF",
}

MODELS_ROOT = Path("/content/trending-models")
ENGINE_ROOT = Path("/content/llama-latest")
OW_ROOT = Path("/content/open-webui")
OW_ENV = OW_ROOT / "venv312"
NVIDIA_LIB = "/usr/lib64-nvidia"
OPENWEBUI_VERSION = "0.12.3"
FALLBACK_LLAMA_TAG = "b12430"

LLAMA_HOST = "127.0.0.1"
OW_HOST = "127.0.0.1"

_status_lock = threading.Lock()
_stop_event = threading.Event()
_child_procs: list[subprocess.Popen] = []


def log_step(msg: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[trending-llm {stamp}] {msg}", flush=True)


def touch_heartbeat() -> None:
    if not HEARTBEAT:
        return
    try:
        Path(HEARTBEAT).parent.mkdir(parents=True, exist_ok=True)
        Path(HEARTBEAT).write_text(str(int(time.time())), encoding="utf-8")
    except OSError:
        pass


def write_status(
    state: str,
    stage: str,
    message: str,
    progress: Optional[float] = None,
    outputs: Optional[dict[str, Any]] = None,
) -> None:
    payload: dict[str, Any] = {
        "state": state,
        "stage": stage,
        "message": message,
    }
    if progress is not None:
        payload["progress"] = max(0.0, min(1.0, float(progress)))
    if outputs:
        payload["outputs"] = outputs
    with _status_lock:
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATUS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, STATUS_FILE)


def _terminate_children() -> None:
    for proc in list(_child_procs):
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
    _child_procs.clear()


def fail(stage: str, message: str) -> None:
    log_step(f"FAILED ({stage}): {message}")
    write_status("failed", stage, message)
    _terminate_children()
    sys.exit(1)


def _on_sigterm(_signum: int, _frame: Any) -> None:
    _stop_event.set()
    _terminate_children()
    sys.exit(0)


def _heartbeat_loop(setup_done: threading.Event, api_key_box: list[str]) -> None:
    while not _stop_event.wait(30.0):
        if not setup_done.is_set():
            touch_heartbeat()
            continue
        try:
            import requests

            headers = {"Authorization": f"Bearer {api_key_box[0]}"} if api_key_box else {}
            r = requests.get(f"http://{LLAMA_HOST}:{LLAMA_PORT}/slots", headers=headers, timeout=4)
            if r.ok and any(s.get("is_processing") for s in r.json()):
                touch_heartbeat()
        except Exception:
            pass


def query_gpu() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"available": False, "name": "CPU", "vram_used_mb": 0.0, "vram_total_mb": 0.0}
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        parts = [p.strip() for p in out.splitlines()[0].split(",")]
        return {
            "available": True,
            "name": parts[0],
            "vram_used_mb": float(parts[1]),
            "vram_total_mb": float(parts[2]),
        }
    except Exception:
        return {"available": False, "name": "CPU", "vram_used_mb": 0.0, "vram_total_mb": 0.0}


def ensure_host_packages() -> None:
    missing = []
    for mod, pkg in [
        ("requests", "requests"),
        ("huggingface_hub", "huggingface_hub"),
        ("hf_transfer", "hf_transfer"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        uv = shutil.which("uv")
        if uv:
            subprocess.run([uv, "pip", "install", "--system", "-q", *missing], check=True)
        else:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)


def ensure_openwebui_installed() -> None:
    OW_ROOT.mkdir(parents=True, exist_ok=True)
    ok_file = OW_ROOT / "install.ok"
    ow_bin = OW_ENV / "bin" / "open-webui"
    if ok_file.is_file() and ok_file.read_text().strip() == OPENWEBUI_VERSION and ow_bin.is_file():
        log_step(f"Open WebUI {OPENWEBUI_VERSION} already installed.")
        return

    uv = shutil.which("uv") or "/usr/local/bin/uv"
    if not Path(uv).exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
        uv = shutil.which("uv") or "/usr/local/bin/uv"

    py312 = "/usr/bin/python3.12"
    if not Path(py312).exists():
        py312 = shutil.which("python3.12") or sys.executable

    venv_py = str(OW_ENV / "bin" / "python")
    install_log = RUN_DIR / "openwebui_install.log"
    with open(install_log, "w", encoding="utf-8") as log_f:
        if not Path(venv_py).exists():
            log_step("Creating isolated venv for Open WebUI...")
            subprocess.run([uv, "venv", "--python", py312, str(OW_ENV)], check=True, stdout=log_f, stderr=subprocess.STDOUT)
        log_step("Installing CPU PyTorch & Open WebUI...")
        subprocess.run(
            [uv, "pip", "install", "--python", venv_py, "torch", "--index-url", "https://download.pytorch.org/whl/cpu"],
            check=True,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        subprocess.run(
            [uv, "pip", "install", "--python", venv_py, f"open-webui=={OPENWEBUI_VERSION}", "--index-url", "https://pypi.org/simple"],
            check=True,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
    ok_file.write_text(OPENWEBUI_VERSION, encoding="utf-8")
    log_step(f"Open WebUI {OPENWEBUI_VERSION} ready.")


def _llama_env(bindir: Path) -> dict[str, str]:
    libs = sorted({str(bindir)} | {str(p.parent) for p in ENGINE_ROOT.rglob("*.so*")})
    if os.path.isdir(NVIDIA_LIB):
        libs.append(NVIDIA_LIB)
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    ld = ":".join(libs + ([existing] if existing else []))
    return dict(os.environ, LD_LIBRARY_PATH=ld)


def ensure_llama_engine() -> tuple[Path, str]:
    import requests

    ENGINE_ROOT.mkdir(parents=True, exist_ok=True)
    ready_file = ENGINE_ROOT / "ready.json"
    if ready_file.is_file():
        try:
            meta = json.loads(ready_file.read_text(encoding="utf-8"))
            saved_bin = Path(meta["binary"])
            if saved_bin.is_file():
                log_step(f"Reusing llama.cpp {meta['tag']} at {saved_bin}.")
                return saved_bin, meta["tag"]
        except Exception:
            pass

    tag = FALLBACK_LLAMA_TAG
    assets: list[dict[str, Any]] = []
    try:
        r = requests.get(
            "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
            headers={"Accept": "application/vnd.github+json", "User-Agent": "colab-playground"},
            timeout=15,
        )
        if r.ok:
            rel = r.json()
            tag = rel.get("tag_name") or tag
            assets = rel.get("assets") or []
    except Exception:
        pass

    if not assets:
        r = requests.get(
            f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}",
            headers={"Accept": "application/vnd.github+json", "User-Agent": "colab-playground"},
            timeout=15,
        )
        r.raise_for_status()
        assets = r.json().get("assets") or []

    wanted = [
        a
        for a in assets
        if a["name"].endswith("-ubuntu-cuda-12.8-x64.tar.gz")
        or a["name"].endswith("-ubuntu-cuda-12.4-x64.tar.gz")
    ]
    # Prefer 12.8 if available, else 12.4
    c128 = [a for a in wanted if "cuda-12.8" in a["name"]]
    chosen = c128 if c128 else wanted
    if not chosen:
        # Fallback to any ubuntu-x64 cuda tar.gz
        chosen = [a for a in assets if "ubuntu" in a["name"] and "cuda" in a["name"] and a["name"].endswith(".tar.gz")]
    if not chosen:
        raise RuntimeError(f"No prebuilt Ubuntu CUDA llama.cpp archive found in release {tag}")

    log_step(f"Downloading llama.cpp {tag} ({', '.join(a['name'] for a in chosen)})...")
    for a in chosen:
        archive = ENGINE_ROOT / a["name"]
        if not archive.exists():
            urllib.request.urlretrieve(a["browser_download_url"], archive)
        with tarfile.open(archive) as tf:
            tf.extractall(ENGINE_ROOT, filter="data")
        try:
            archive.unlink(missing_ok=True)
        except OSError:
            pass

    binary = next(ENGINE_ROOT.rglob("llama-server"))
    res = subprocess.run(
        [str(binary), "--help"],
        env=_llama_env(binary.parent),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if res.returncode != 0:
        raise RuntimeError(f"llama-server --help failed: {res.stdout[-1500:]}")

    ready_file.write_text(json.dumps({"binary": str(binary), "tag": tag}), encoding="utf-8")
    log_step(f"llama.cpp {tag} ready ({binary}).")
    return binary, tag


def _strip_shard_suffix(stem: str) -> str:
    m = re.match(r"^(.*)-\d{5}-of-\d{5}$", stem)
    return m.group(1) if m else stem


def _quant_label(path: str) -> Optional[str]:
    base = path.rsplit("/", 1)[-1]
    if not base.lower().endswith(".gguf"):
        return None
    tokens = re.split(r"[-.]", _strip_shard_suffix(base[:-5]))
    for i in range(len(tokens) - 1, -1, -1):
        u = tokens[i].upper()
        if u in ("F16", "F32", "BF16") or u.startswith("MXFP4") or (
            any(u.startswith(p) for p in ("IQ", "TQ", "Q")) and any(ch.isdigit() for ch in u)
        ):
            if i > 0 and tokens[i - 1].lower() == "ud":
                return "UD-" + u
            return u
    return None


def plan_gguf(repo_id: str, vram_mib: float) -> dict[str, Any]:
    from huggingface_hub import HfApi

    token_arg: Any = HF_TOKEN if HF_TOKEN else False
    info = HfApi(token=token_arg).model_info(repo_id, files_metadata=True)
    by_label: dict[str, dict[str, Any]] = {}
    mmprojs: list[tuple[str, int]] = []

    for s in info.siblings:
        rf = s.rfilename
        sz = s.size or 0
        if not rf.lower().endswith(".gguf"):
            continue
        if "mmproj" in rf.rsplit("/", 1)[-1].lower():
            mmprojs.append((rf, sz))
            continue
        label = _quant_label(rf)
        if not label:
            continue
        entry = by_label.setdefault(label, {"quant": label, "files": [], "size": 0})
        entry["files"].append(rf)
        entry["size"] += sz

    if not by_label:
        raise RuntimeError(f"Repository {repo_id} has no GGUF model files")

    for v in by_label.values():
        v["files"].sort()

    # Pick mmproj if present (prefer F16 / BF16)
    mmproj_file = None
    mmproj_size = 0
    if mmprojs:
        mmprojs.sort(key=lambda x: (0 if "f16" in x[0].lower() or "bf16" in x[0].lower() else 1, -x[1]))
        mmproj_file, mmproj_size = mmprojs[0]

    variants = sorted(by_label.values(), key=lambda v: (-v["size"], v["quant"]))
    full_prec = ("F16", "BF16", "F32")
    quantized = [v for v in variants if v["quant"] not in full_prec]
    pool = quantized if quantized else variants

    chosen = None
    if QUANT_PREF != "auto":
        for v in variants:
            if v["quant"].upper() == QUANT_PREF.upper():
                chosen = v
                break
        if chosen is None:
            log_step(f"Requested quant {QUANT_PREF} not found in {repo_id}; falling back to auto.")

    if chosen is None:
        # Leave ~18% VRAM headroom (at least 3.5 GiB) for KV cache and CUDA compute buffers.
        budget_bytes = max(0.0, (vram_mib * 0.82 - 1500.0)) * 1024 * 1024 if vram_mib > 0 else 0.0
        for v in pool:
            if v["size"] + mmproj_size <= budget_bytes:
                chosen = v
                break
        if chosen is None:
            chosen = pool[-1]  # smallest available quant

    files = list(chosen["files"]) + ([mmproj_file] if mmproj_file else [])
    total_size = chosen["size"] + mmproj_size
    return {
        "quant": chosen["quant"],
        "files": files,
        "model_first_file": chosen["files"][0],
        "mmproj_file": mmproj_file,
        "total_size": total_size,
    }


def _dir_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def download_files(repo_id: str, files: list[str], total_expected: int) -> Path:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    token_arg: Any = HF_TOKEN if HF_TOKEN else False
    if not HF_TOKEN:
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

    safe_name = repo_id.replace("/", "__")
    local_dir = MODELS_ROOT / safe_name
    local_dir.mkdir(parents=True, exist_ok=True)
    hf_tmp = local_dir / ".hf_home"
    os.environ["HF_HOME"] = str(hf_tmp)

    total_gb = max(0.01, total_expected / 1e9)
    stop_mon = threading.Event()
    t0 = time.time()

    def _monitor() -> None:
        last_b = _dir_bytes(local_dir)
        last_t = time.time()
        while not stop_mon.wait(6.0):
            now = time.time()
            cur_b = _dir_bytes(local_dir)
            dt = max(0.1, now - last_t)
            mbps = max(0.0, (cur_b - last_b) / (1024 * 1024) / dt)
            frac = min(0.99, cur_b / max(1, total_expected))
            msg = f"{cur_b / 1e9:.1f} / {total_gb:.1f} GB ({mbps:.0f} MB/s)"
            write_status("running", "download", msg, progress=frac)
            log_step(f"Download: {msg}")
            last_b, last_t = cur_b, now

    mon = threading.Thread(target=_monitor, daemon=True)
    mon.start()
    try:
        for idx, rf in enumerate(files, start=1):
            log_step(f"Fetching [{idx}/{len(files)}] {rf}...")
            try:
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
                hf_hub_download(repo_id=repo_id, filename=rf, local_dir=str(local_dir), token=token_arg)
            except Exception as exc:
                log_step(f"hf_transfer fallback ({exc})...")
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
                hf_hub_download(repo_id=repo_id, filename=rf, local_dir=str(local_dir), token=token_arg)
    finally:
        stop_mon.set()
        mon.join(timeout=2.0)

    shutil.rmtree(local_dir / ".cache", ignore_errors=True)
    shutil.rmtree(hf_tmp, ignore_errors=True)
    write_status("running", "download", f"Downloaded {total_gb:.1f} GB in {int(time.time() - t0)}s", progress=1.0)
    return local_dir


def _http_ok(url: str, timeout: float = 2.5) -> bool:
    import requests

    try:
        return requests.get(url, timeout=timeout).ok
    except Exception:
        return False


def load_or_create_access() -> dict[str, str]:
    OW_ROOT.mkdir(parents=True, exist_ok=True)
    access_file = OW_ROOT / "access.json"
    if access_file.is_file():
        access = json.loads(access_file.read_text(encoding="utf-8"))
        if ADMIN_PASSWORD:
            access["password"] = ADMIN_PASSWORD
        if ADMIN_EMAIL:
            access["email"] = ADMIN_EMAIL
        return access
    access = {
        "email": ADMIN_EMAIL or "rick@example.com",
        "password": ADMIN_PASSWORD or secrets.token_urlsafe(20),
        "secret": secrets.token_urlsafe(32),
    }
    fd = os.open(access_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(access, f)
    return access


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _on_sigterm)
    signal.signal(signal.SIGINT, _on_sigterm)

    if OW_PORT == 8080 or LLAMA_PORT == 8080:
        fail("setup", "VM port 8080 is reserved by Colab; choose another port.")
    if OW_PORT == LLAMA_PORT:
        fail("setup", "Open WebUI port and llama.cpp API port must be different.")

    repo_id = CUSTOM_REPO if CUSTOM_REPO else PRESET_REPOS.get(PRESET, "")
    if not repo_id or "/" not in repo_id:
        fail("setup", f"Invalid model repository: {repo_id!r}")
    alias = CUSTOM_REPO.split("/")[-1] if CUSTOM_REPO else PRESET

    setup_done = threading.Event()
    api_key_box: list[str] = []
    threading.Thread(target=_heartbeat_loop, args=(setup_done, api_key_box), daemon=True).start()
    touch_heartbeat()

    stage = "setup"
    try:
        write_status("running", "setup", "Installing helper packages, llama.cpp & Open WebUI...")
        ensure_host_packages()
        gpu = query_gpu()
        log_step(f"Detected GPU: {gpu['name']} ({gpu['vram_total_mb']:.0f} MiB)")

        stage = "plan"
        write_status("running", "plan", f"Inspecting GGUF files in {repo_id}...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            fut_ow = pool.submit(ensure_openwebui_installed)
            fut_eng = pool.submit(ensure_llama_engine)
            plan = plan_gguf(repo_id, gpu["vram_total_mb"])
            log_step(
                f"Selected {plan['quant']} ({plan['total_size']/1e9:.2f} GB)"
                + (f" + vision mmproj {plan['mmproj_file']}" if plan["mmproj_file"] else "")
            )
            stage = "download"
            write_status(
                "running",
                "download",
                f"Downloading {alias} ({plan['quant']}, {plan['total_size']/1e9:.1f} GB)...",
                progress=0.0,
            )
            local_dir = download_files(repo_id, plan["files"], plan["total_size"])
            binary, llama_tag = fut_eng.result()
            fut_ow.result()

        stage = "server"
        write_status("running", "server", f"Starting llama-server ({alias}) & Open WebUI...")
        subprocess.run(["pkill", "-f", f"llama-server.*--port {LLAMA_PORT}"], check=False)
        subprocess.run(["pkill", "-f", f"open-webui.*--port {OW_PORT}"], check=False)
        time.sleep(1.0)

        api_key = secrets.token_urlsafe(24)
        api_key_box.append(api_key)
        keyfile = RUN_DIR / "api.key"
        keyfile.write_text(api_key, encoding="utf-8")
        keyfile.chmod(0o600)

        model_path = local_dir / plan["model_first_file"]
        cmd = [
            str(binary),
            "--model",
            str(model_path),
            "--host",
            LLAMA_HOST,
            "--port",
            str(LLAMA_PORT),
            "--alias",
            alias,
            "--api-key-file",
            str(keyfile),
            "--ctx-size",
            str(CTX_SIZE),
            "--n-gpu-layers",
            "99",
            "--jinja",
        ]
        if plan["mmproj_file"]:
            cmd.extend(["--mmproj", str(local_dir / plan["mmproj_file"])])

        llama_log = RUN_DIR / "llama-server.log"
        llama_proc = subprocess.Popen(
            cmd,
            env=_llama_env(binary.parent),
            stdin=subprocess.DEVNULL,
            stdout=open(llama_log, "w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _child_procs.append(llama_proc)

        access = load_or_create_access()
        ow_env = dict(
            os.environ,
            DATA_DIR=str(OW_ROOT / "data"),
            WEBUI_SECRET_KEY=access["secret"],
            WEBUI_ADMIN_EMAIL=access["email"],
            WEBUI_ADMIN_PASSWORD=access["password"],
            WEBUI_ADMIN_NAME="Admin",
            ENABLE_SIGNUP="false",
            ENABLE_PERSISTENT_CONFIG="false",
            ENABLE_OLLAMA_API="false",
            OPENAI_API_BASE_URL=f"http://{LLAMA_HOST}:{LLAMA_PORT}/v1",
            OPENAI_API_KEY=api_key,
            DEFAULT_MODELS=alias,
            OFFLINE_MODE="true",
            HF_HUB_OFFLINE="1",
            SCARF_NO_ANALYTICS="true",
            DO_NOT_TRACK="true",
            ANONYMIZED_TELEMETRY="false",
        )
        ow_log = RUN_DIR / "openwebui.log"
        ow_proc = subprocess.Popen(
            [str(OW_ENV / "bin" / "open-webui"), "serve", "--host", OW_HOST, "--port", str(OW_PORT)],
            env=ow_env,
            cwd=str(OW_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=open(ow_log, "w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _child_procs.append(ow_proc)

        for _ in range(150):
            if llama_proc.poll() is not None:
                tail = llama_log.read_text(errors="replace")[-1500:]
                raise RuntimeError(f"llama-server exited early: {tail}")
            if ow_proc.poll() is not None:
                tail = ow_log.read_text(errors="replace")[-1500:]
                raise RuntimeError(f"Open WebUI exited early: {tail}")
            if _http_ok(f"http://{LLAMA_HOST}:{LLAMA_PORT}/health") and _http_ok(f"http://{OW_HOST}:{OW_PORT}/health"):
                break
            time.sleep(2.0)
        else:
            raise TimeoutError("Timed out waiting for llama-server and Open WebUI /health")

        gpu_now = query_gpu()
        outputs: dict[str, Any] = {
            "model_id": alias,
            "repo": repo_id,
            "quant": plan["quant"],
            "vision_mmproj": plan["mmproj_file"] or "none",
            "ctx_size": CTX_SIZE,
            "login_email": access["email"],
            "login_password": access["password"],
            "llama_cpp": llama_tag,
            "gpu": gpu_now["name"],
            "vram_used_mib": int(gpu_now["vram_used_mb"]),
            "vram_total_mib": int(gpu_now["vram_total_mb"]),
        }
        setup_done.set()
        write_status(
            "ready",
            "server",
            f"{alias} ({plan['quant']}) ready on {gpu_now['name']}",
            progress=1.0,
            outputs=outputs,
        )
        log_step(f"Ready! {alias} ({plan['quant']}) + Open WebUI running.")

        while not _stop_event.wait(10.0):
            if llama_proc.poll() is not None:
                fail("server", f"llama-server exited unexpectedly ({llama_proc.returncode})")
            if ow_proc.poll() is not None:
                fail("server", f"Open WebUI exited unexpectedly ({ow_proc.returncode})")
    except SystemExit:
        raise
    except Exception as exc:
        fail(stage, str(exc))


if __name__ == "__main__":
    main()
