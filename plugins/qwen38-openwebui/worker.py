#!/usr/bin/env python3
"""Qwen3.8-Flash-Next (IQ3_S, ~84 GB) + Open WebUI 0.12.3 worker for Colab Playground.

Adapted from Colab Studio's `qwen38_openwebui_worker.py` (Cells 1-8 of
`Qwen3.8-Flash-Next-IQ3_S-OpenWebUI.ipynb`) for Colab Playground's plugin
protocol (`docs/PLUGINS.md`).
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

ADMIN_EMAIL = (E("PLUGIN_INPUT_ADMIN_EMAIL", "rick@example.com") or "rick@example.com").strip()
ADMIN_PASSWORD = (E("PLUGIN_INPUT_ADMIN_PASSWORD", "") or "").strip()
CTX_SIZE = int(float(E("PLUGIN_INPUT_CTX_SIZE", "32768") or 32768))
OW_PORT = int(float(E("PLUGIN_INPUT_WEBUI_PORT", "8081") or 8081))
LLAMA_PORT = int(float(E("PLUGIN_INPUT_API_PORT", "8000") or 8000))
SKIP_ACCEPTANCE = E("PLUGIN_INPUT_SKIP_ACCEPTANCE", "0") == "1"
ENABLE_CF_TUNNEL = E("PLUGIN_INPUT_CLOUDFLARE_TUNNEL", "0") == "1"
HF_TOKEN = (E("PLUGIN_HF_TOKEN", "") or "").strip()

GGUF_ROOT = Path("/content/qwen38")
MODEL_DIR = GGUF_ROOT / "model"
ENGINE_ROOT = GGUF_ROOT / "engine"
OW_ROOT = Path("/content/open-webui")
OW_ENV = OW_ROOT / "venv312"
CLOUDFLARED_BIN = Path("/content/cloudflared")
STATE_FILE = OW_ROOT / "state.json"
NVIDIA_LIB = "/usr/lib64-nvidia"

REPO_ID = "SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF"
MODEL_REVISION = "90a3350cd800ec1fe0d1660a38414ad9e985782f"
LLAMA_MODEL = "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S"
LLAMA_RELEASE = "b12110"
LLAMA_ARCHIVES = [
    f"llama-{LLAMA_RELEASE}-bin-ubuntu-cuda-12.8-x64.tar.gz",
    f"cudart-llama-{LLAMA_RELEASE}-bin-ubuntu-cuda-12.8-x64.tar.gz",
]

LLAMA_HOST = "127.0.0.1"
OW_HOST = "127.0.0.1"
OPENWEBUI_VERSION = "0.12.3"

_status_lock = threading.Lock()
_stop_event = threading.Event()
_child_procs: list[subprocess.Popen] = []


def log_step(msg: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[qwen38-openwebui {stamp}] {msg}", flush=True)


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


def fail(stage: str, message: str) -> None:
    log_step(f"FAILED ({stage}): {message}")
    write_status("failed", stage, message)
    _terminate_children()
    sys.exit(1)


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


def _on_sigterm(_signum: int, _frame: Any) -> None:
    _stop_event.set()
    _terminate_children()
    sys.exit(0)


def _heartbeat_loop(setup_done: threading.Event, llama_api_key_box: list[str]) -> None:
    while not _stop_event.wait(30.0):
        if not setup_done.is_set():
            touch_heartbeat()
            continue
        # Once ready, touch heartbeat whenever llama-server has an active slot.
        try:
            import requests

            headers = {}
            if llama_api_key_box:
                headers["Authorization"] = f"Bearer {llama_api_key_box[0]}"
            r = requests.get(f"http://{LLAMA_HOST}:{LLAMA_PORT}/slots", headers=headers, timeout=4)
            if r.ok and any(s.get("is_processing") for s in r.json()):
                touch_heartbeat()
        except Exception:
            pass


def query_gpu_summary() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"available": False, "name": "CPU", "vram_used_mb": 0.0, "vram_total_mb": 0.0}
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
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
            "gpu_util_percent": float(parts[3]),
        }
    except Exception as exc:
        return {"available": False, "name": f"Error: {exc}", "vram_used_mb": 0.0, "vram_total_mb": 0.0}


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
        log_step(f"Installing helper packages: {', '.join(missing)}...")
        uv = shutil.which("uv")
        if uv:
            subprocess.run([uv, "pip", "install", "--system", "-q", *missing], check=True)
        else:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)


def ensure_cloudflared() -> Path:
    if not CLOUDFLARED_BIN.is_file():
        log_step("Downloading cloudflared binary...")
        urllib.request.urlretrieve(
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            CLOUDFLARED_BIN,
        )
        CLOUDFLARED_BIN.chmod(0o700)
    subprocess.run(
        [str(CLOUDFLARED_BIN), "--version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_step("cloudflared ready.")
    return CLOUDFLARED_BIN


def ensure_openwebui_installed() -> None:
    OW_ROOT.mkdir(parents=True, exist_ok=True)
    ok_file = OW_ROOT / "install.ok"
    ow_bin = OW_ENV / "bin" / "open-webui"
    if ok_file.is_file() and ok_file.read_text().strip() == OPENWEBUI_VERSION and ow_bin.is_file():
        log_step(f"Open WebUI {OPENWEBUI_VERSION} already installed in {OW_ENV}.")
        return

    uv = shutil.which("uv") or "/usr/local/bin/uv"
    if not Path(uv).exists():
        log_step("Installing uv...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
        uv = shutil.which("uv") or "/usr/local/bin/uv"

    py312 = "/usr/bin/python3.12"
    if not Path(py312).exists():
        py312 = shutil.which("python3.12") or sys.executable

    venv_py = str(OW_ENV / "bin" / "python")
    install_log = RUN_DIR / "openwebui_install.log"
    with open(install_log, "w", encoding="utf-8") as log_f:
        if not Path(venv_py).exists():
            log_step("Creating isolated Python 3.12 venv for Open WebUI...")
            subprocess.run(
                [uv, "venv", "--python", py312, str(OW_ENV)],
                check=True,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )

        log_step("Installing CPU PyTorch into Open WebUI venv312...")
        subprocess.run(
            [uv, "pip", "install", "--python", venv_py, "torch", "--index-url", "https://download.pytorch.org/whl/cpu"],
            check=True,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )

        log_step(f"Installing open-webui=={OPENWEBUI_VERSION} into venv312...")
        subprocess.run(
            [uv, "pip", "install", "--python", venv_py, f"open-webui=={OPENWEBUI_VERSION}", "--index-url", "https://pypi.org/simple"],
            check=True,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )

    ok_file.write_text(OPENWEBUI_VERSION, encoding="utf-8")
    log_step(f"Open WebUI {OPENWEBUI_VERSION} installation complete.")


def _llama_env() -> dict[str, str]:
    libs = sorted({str(p.parent) for p in ENGINE_ROOT.rglob("*.so*")})
    if os.path.isdir(NVIDIA_LIB):
        libs.append(NVIDIA_LIB)
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    ld = ":".join(libs + ([existing] if existing else []))
    return dict(os.environ, LD_LIBRARY_PATH=ld)


def ensure_llama_engine() -> Path:
    ENGINE_ROOT.mkdir(parents=True, exist_ok=True)
    ready_file = ENGINE_ROOT / "ready.ok"
    if ready_file.is_file():
        saved_bin = Path(ready_file.read_text().strip())
        if saved_bin.is_file():
            log_step(f"llama.cpp {LLAMA_RELEASE} engine already ready at {saved_bin}.")
            return saved_bin

    log_step(f"Downloading & extracting llama.cpp {LLAMA_RELEASE} CUDA 12.8...")
    for name in LLAMA_ARCHIVES:
        archive = ENGINE_ROOT / name
        if not archive.exists():
            url = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_RELEASE}/{name}"
            urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive) as tf:
            tf.extractall(ENGINE_ROOT, filter="data")

    binary = next(ENGINE_ROOT.rglob("llama-server"))
    res = subprocess.run(
        [str(binary), "--help"],
        env=_llama_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if res.returncode != 0:
        raise RuntimeError(f"llama-server --help failed ({res.returncode}): {res.stdout[-2000:]}")

    ready_file.write_text(str(binary), encoding="utf-8")
    for name in LLAMA_ARCHIVES:
        try:
            (ENGINE_ROOT / name).unlink(missing_ok=True)
        except OSError:
            pass
    log_step(f"llama.cpp {LLAMA_RELEASE} CUDA 12.8 engine verified ({binary}).")
    return binary


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


def ensure_model_downloaded() -> Path:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    token_arg: Any = HF_TOKEN if HF_TOKEN else False
    if not HF_TOKEN:
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    GGUF_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    ok_file = GGUF_ROOT / "download.ok"
    rev_file = GGUF_ROOT / "revision.txt"
    if ok_file.is_file() and rev_file.is_file() and rev_file.read_text().strip() == MODEL_REVISION:
        shards = sorted((MODEL_DIR / "IQ3_S").glob("*-00001-of-00002.gguf"))
        if shards and len(list((MODEL_DIR / "IQ3_S").glob("*.gguf"))) == 2:
            log_step(f"Model {LLAMA_MODEL} ({MODEL_REVISION[:10]}) already verified on disk.")
            write_status("running", "download", f"{LLAMA_MODEL} already on disk", progress=1.0)
            return shards[0]

    info = HfApi(token=token_arg).model_info(REPO_ID, revision=MODEL_REVISION, files_metadata=True)
    files = [f for f in info.siblings if f.rfilename.startswith("IQ3_S/") and f.rfilename.endswith(".gguf")]
    if len(files) != 2:
        raise RuntimeError(f"Expected 2 IQ3_S shards, found {len(files)}")

    total_expected = sum(f.size or 0 for f in files)
    total_gb = round(total_expected / 1e9, 2)

    free_bytes = shutil.disk_usage(str(GGUF_ROOT)).free
    already_bytes = sum(
        (MODEL_DIR / f.rfilename).stat().st_size
        for f in files
        if (MODEL_DIR / f.rfilename).is_file()
    )
    needed_bytes = max(0, total_expected - already_bytes)
    if free_bytes < needed_bytes + 4 * (1024 ** 3):
        raise RuntimeError(
            f"Insufficient disk space on /content: {free_bytes / (1024**3):.1f} GiB free, "
            f"need {(needed_bytes / (1024**3)):.1f} GiB for {LLAMA_MODEL} (enable High-RAM)"
        )

    log_step(f"Downloading {LLAMA_MODEL} ({total_gb} GB, 2 shards) via hf_transfer...")
    write_status("running", "download", f"0.0 / {total_gb:.1f} GB", progress=0.0)

    hf_cache_tmp = GGUF_ROOT / ".hf_home"
    os.environ["HF_HOME"] = str(hf_cache_tmp)

    stop_monitor = threading.Event()
    t0 = time.time()

    def _progress_monitor() -> None:
        last_bytes = _dir_bytes(MODEL_DIR)
        last_t = time.time()
        while not stop_monitor.wait(8.0):
            now = time.time()
            cur_bytes = _dir_bytes(MODEL_DIR)
            dt = max(0.1, now - last_t)
            mbps = max(0.0, (cur_bytes - last_bytes) / (1024 * 1024) / dt)
            frac = min(0.99, cur_bytes / max(1, total_expected))
            msg = f"{cur_bytes / 1e9:.1f} / {total_gb:.1f} GB ({mbps:.0f} MB/s)"
            write_status("running", "download", msg, progress=frac)
            log_step(f"Model download: {msg} (elapsed {int(now - t0)}s)")
            last_bytes = cur_bytes
            last_t = now

    mon_thread = threading.Thread(target=_progress_monitor, daemon=True)
    mon_thread.start()
    try:
        try:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            for idx, f in enumerate(files, start=1):
                target = MODEL_DIR / f.rfilename
                if target.is_file() and target.stat().st_size == f.size:
                    log_step(f"Shard [{idx}/2] {f.rfilename} already complete.")
                    continue
                log_step(f"Fetching shard [{idx}/2] {f.rfilename} ({(f.size or 0) / 1e9:.2f} GB)...")
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=f.rfilename,
                    revision=MODEL_REVISION,
                    local_dir=str(MODEL_DIR),
                    token=token_arg,
                )
        except Exception as exc:
            log_step(f"hf_transfer issue ({exc}); falling back to snapshot_download...")
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
            snapshot_download(
                REPO_ID,
                revision=MODEL_REVISION,
                allow_patterns=[f.rfilename for f in files],
                local_dir=str(MODEL_DIR),
                max_workers=2,
                token=token_arg,
            )
    finally:
        stop_monitor.set()
        mon_thread.join(timeout=2.0)

    for f in files:
        p = MODEL_DIR / f.rfilename
        if not p.is_file() or p.stat().st_size != f.size:
            got = p.stat().st_size if p.is_file() else -1
            raise RuntimeError(f"Shard size mismatch for {f.rfilename}: got {got}, expected {f.size}")

    shutil.rmtree(MODEL_DIR / ".cache", ignore_errors=True)
    shutil.rmtree(hf_cache_tmp, ignore_errors=True)

    rev_file.write_text(MODEL_REVISION, encoding="utf-8")
    ok_file.write_text("ok", encoding="utf-8")
    elapsed = int(time.time() - t0)
    write_status("running", "download", f"Downloaded {total_gb:.1f} GB in {elapsed}s", progress=1.0)
    log_step(f"Model download & verification succeeded in {elapsed}s.")
    return next((MODEL_DIR / "IQ3_S").glob("*-00001-of-00002.gguf"))


def _http_ok(url: str, timeout: float = 2.5) -> bool:
    import requests

    try:
        return requests.get(url, timeout=timeout).ok
    except Exception:
        return False


def _pkill_pattern(pattern: str) -> None:
    subprocess.run(["pkill", "-f", pattern], check=False)


def start_llama_server(binary: Path, shard: Path) -> tuple[str, Optional[subprocess.Popen]]:
    import requests

    keyfile = GGUF_ROOT / "api.key"
    if keyfile.is_file() and keyfile.read_text().strip():
        llama_api_key = keyfile.read_text().strip()
    else:
        llama_api_key = secrets.token_urlsafe(32)
        keyfile.write_text(llama_api_key, encoding="utf-8")
    keyfile.chmod(0o600)

    if _http_ok(f"http://{LLAMA_HOST}:{LLAMA_PORT}/health"):
        try:
            r = requests.get(
                f"http://{LLAMA_HOST}:{LLAMA_PORT}/v1/models",
                headers={"Authorization": f"Bearer {llama_api_key}"},
                timeout=5,
            )
            if r.ok and r.json()["data"][0]["id"] == LLAMA_MODEL:
                log_step(f"Reusing healthy llama-server on {LLAMA_HOST}:{LLAMA_PORT}.")
                return llama_api_key, None
        except Exception:
            pass
        _pkill_pattern(f"llama-server.*--port {LLAMA_PORT}")
        time.sleep(1.5)

    log_path = RUN_DIR / "llama-server.log"
    log_f = open(log_path, "w", encoding="utf-8")
    cmd = [
        str(binary),
        "--model",
        str(shard),
        "--host",
        LLAMA_HOST,
        "--port",
        str(LLAMA_PORT),
        "--alias",
        LLAMA_MODEL,
        "--api-key-file",
        str(keyfile),
        "--ctx-size",
        str(CTX_SIZE),
        "--parallel",
        "1",
        "--n-gpu-layers",
        "99",
        "--load-mode",
        "mmap",
        "--lazy-mode",
        "on",
        "--jinja",
        "--reasoning",
        "off",
    ]
    proc = subprocess.Popen(
        cmd,
        env=_llama_env(),
        stdin=subprocess.DEVNULL,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _child_procs.append(proc)
    log_step(f"Started llama-server (PID {proc.pid}) for {LLAMA_MODEL} on {LLAMA_HOST}:{LLAMA_PORT}.")
    return llama_api_key, proc


def load_or_create_access() -> dict[str, str]:
    OW_ROOT.mkdir(parents=True, exist_ok=True)
    access_file = OW_ROOT / "access.json"
    if access_file.is_file():
        access = json.loads(access_file.read_text(encoding="utf-8"))
        if ADMIN_PASSWORD and access.get("password") != ADMIN_PASSWORD:
            access["password"] = ADMIN_PASSWORD
        if ADMIN_EMAIL and access.get("email") != ADMIN_EMAIL:
            access["email"] = ADMIN_EMAIL
        return access

    access = {
        "email": ADMIN_EMAIL or "rick@example.com",
        "password": ADMIN_PASSWORD or secrets.token_urlsafe(24),
        "secret": secrets.token_urlsafe(32),
    }
    fd = os.open(access_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(access, f)
    return access


def start_openwebui(llama_api_key: str) -> tuple[dict[str, str], Optional[subprocess.Popen]]:
    access = load_or_create_access()
    ow_login = access["email"]
    web_password = access["password"]
    ow_secret = access["secret"]

    if _http_ok(f"http://{OW_HOST}:{OW_PORT}/health"):
        log_step(f"Reusing healthy Open WebUI on {OW_HOST}:{OW_PORT} ({ow_login}).")
        return access, None

    _pkill_pattern(f"open-webui.*--port {OW_PORT}")
    time.sleep(1.0)

    ow_env = dict(
        os.environ,
        DATA_DIR=str(OW_ROOT / "data"),
        WEBUI_SECRET_KEY=ow_secret,
        WEBUI_ADMIN_EMAIL=ow_login,
        WEBUI_ADMIN_PASSWORD=web_password,
        WEBUI_ADMIN_NAME="Admin",
        ENABLE_SIGNUP="false",
        ENABLE_PERSISTENT_CONFIG="false",
        ENABLE_OLLAMA_API="false",
        OPENAI_API_BASE_URL=f"http://{LLAMA_HOST}:{LLAMA_PORT}/v1",
        OPENAI_API_KEY=llama_api_key,
        DEFAULT_MODELS=LLAMA_MODEL,
        OFFLINE_MODE="true",
        HF_HUB_OFFLINE="1",
        RAG_EMBEDDING_ENGINE="openai",
        RAG_EMBEDDING_MODEL_AUTO_UPDATE="false",
        WHISPER_MODEL_AUTO_UPDATE="false",
        SCARF_NO_ANALYTICS="true",
        DO_NOT_TRACK="true",
        ANONYMIZED_TELEMETRY="false",
    )
    log_f = open(RUN_DIR / "openwebui.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [str(OW_ENV / "bin" / "open-webui"), "serve", "--host", OW_HOST, "--port", str(OW_PORT)],
        env=ow_env,
        cwd=str(OW_ROOT),
        stdin=subprocess.DEVNULL,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _child_procs.append(proc)
    log_step(f"Started Open WebUI {OPENWEBUI_VERSION} (PID {proc.pid}) on {OW_HOST}:{OW_PORT} for {ow_login}.")
    return access, proc


def start_cloudflare_tunnel() -> Optional[str]:
    _pkill_pattern(f"cloudflared.*http://{OW_HOST}:{OW_PORT}")
    time.sleep(1.0)
    tunnel_log_path = RUN_DIR / "cloudflared.log"
    tunnel_f = open(tunnel_log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            str(CLOUDFLARED_BIN),
            "tunnel",
            "--url",
            f"http://{OW_HOST}:{OW_PORT}",
            "--no-autoupdate",
            "--protocol",
            "http2",
        ],
        stdin=subprocess.DEVNULL,
        stdout=tunnel_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _child_procs.append(proc)
    public_url = None
    for _ in range(60):
        log_txt = tunnel_log_path.read_text(errors="replace")
        for m in re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", log_txt):
            if "api.trycloudflare.com" not in m:
                public_url = m
                break
        if public_url:
            break
        if proc.poll() is not None:
            log_step(f"cloudflared exited early: {log_txt[-1000:]}")
            return None
        time.sleep(1.0)
    if public_url:
        for _ in range(20):
            if _http_ok(f"{public_url}/health", timeout=5.0):
                break
            time.sleep(1.0)
        log_step(f"Cloudflare public URL ready: {public_url}")
    return public_url


def verify_services(
    llama_api_key: str,
    access: dict[str, str],
    llama_proc: Optional[subprocess.Popen],
    ow_proc: Optional[subprocess.Popen],
) -> str:
    import requests

    write_status("running", "verify", "Waiting for llama-server and Open WebUI health checks...")
    log_step(f"Waiting for llama-server ({LLAMA_PORT}) and Open WebUI ({OW_PORT}) /health...")
    for _ in range(150):
        if llama_proc is not None and llama_proc.poll() is not None:
            tail = (RUN_DIR / "llama-server.log").read_text(errors="replace")[-2000:]
            raise RuntimeError(f"llama-server exited early: {tail}")
        if ow_proc is not None and ow_proc.poll() is not None:
            tail = (RUN_DIR / "openwebui.log").read_text(errors="replace")[-2000:]
            raise RuntimeError(f"Open WebUI exited early: {tail}")
        if _http_ok(f"http://{LLAMA_HOST}:{LLAMA_PORT}/health") and _http_ok(f"http://{OW_HOST}:{OW_PORT}/health"):
            break
        time.sleep(2.0)
    else:
        raise TimeoutError("Timed out waiting for llama-server and Open WebUI /health")

    api_base = f"http://{LLAMA_HOST}:{LLAMA_PORT}/v1"
    headers = {"Authorization": f"Bearer {llama_api_key}"}

    r_models = requests.get(f"{api_base}/models", headers=headers, timeout=10)
    r_models.raise_for_status()
    got_model_id = r_models.json()["data"][0]["id"]
    if got_model_id != LLAMA_MODEL:
        raise RuntimeError(f"Model ID mismatch: expected {LLAMA_MODEL}, got {got_model_id}")

    stream_reply = "skipped"
    if not SKIP_ACCEPTANCE:
        write_status("running", "verify", "Running multi-turn SSE warmup check...")
        log_step("Running multi-turn SSE streaming acceptance test (`蓝鲸742`)...")
        prompt = [
            {"role": "user", "content": "记住验证码是 蓝鲸742。只回复收到。"},
            {"role": "assistant", "content": "收到。"},
            {"role": "user", "content": "刚才验证码是什么？只输出验证码。"},
        ]
        r_chat = requests.post(
            f"{api_base}/chat/completions",
            headers=headers,
            json={
                "model": LLAMA_MODEL,
                "messages": prompt,
                "stream": True,
                "max_tokens": 64,
                "temperature": 0,
            },
            stream=True,
            timeout=240,
        )
        r_chat.raise_for_status()
        pieces: list[str] = []
        for line in r_chat.iter_lines():
            if line.startswith(b"data: ") and line != b"data: [DONE]":
                delta = json.loads(line[6:])["choices"][0]["delta"]
                pieces.append(delta.get("content") or "")
        stream_reply = "".join(pieces).strip()
        log_step(f"STREAM_MULTI_TURN: {stream_reply}")
        if "蓝鲸742" not in stream_reply:
            raise RuntimeError(f"Acceptance test failed: expected '蓝鲸742', got {stream_reply!r}")

    ow_base = f"http://{OW_HOST}:{OW_PORT}"
    r_signin = requests.post(
        f"{ow_base}/api/v1/auths/signin",
        json={"email": access["email"], "password": access["password"]},
        timeout=20,
    )
    r_signin.raise_for_status()
    ow_token = r_signin.json()["token"]

    r_ow_models = requests.get(
        f"{ow_base}/api/models",
        headers={"Authorization": f"Bearer {ow_token}"},
        timeout=30,
    )
    r_ow_models.raise_for_status()
    if LLAMA_MODEL not in r_ow_models.text:
        raise RuntimeError(f"Open WebUI /api/models did not list {LLAMA_MODEL}")

    log_step("Verification passed: IQ3_S / 32K / SSE / Open WebUI auth & model list OK.")
    return stream_reply


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _on_sigterm)
    signal.signal(signal.SIGINT, _on_sigterm)

    if OW_PORT == 8080 or LLAMA_PORT == 8080:
        fail("setup", "VM port 8080 is reserved by Colab; choose another port.")
    if OW_PORT == LLAMA_PORT:
        fail("setup", "Open WebUI port and llama.cpp API port must be different.")

    setup_done = threading.Event()
    llama_api_key_box: list[str] = []
    threading.Thread(target=_heartbeat_loop, args=(setup_done, llama_api_key_box), daemon=True).start()
    touch_heartbeat()

    stage = "setup"
    try:
        write_status("running", "setup", "Installing helper packages, llama.cpp b12110 & Open WebUI 0.12.3...")
        ensure_host_packages()
        GGUF_ROOT.mkdir(parents=True, exist_ok=True)
        OW_ROOT.mkdir(parents=True, exist_ok=True)

        log_step("Starting parallel setup: Open WebUI 0.12.3 + llama.cpp b12110 + Qwen3.8 IQ3_S (~83.9 GB)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            fut_cf = pool.submit(ensure_cloudflared) if ENABLE_CF_TUNNEL else None
            fut_ow = pool.submit(ensure_openwebui_installed)
            fut_eng = pool.submit(ensure_llama_engine)
            fut_mdl = pool.submit(ensure_model_downloaded)

            if fut_cf is not None:
                fut_cf.result()
            binary = fut_eng.result()
            fut_ow.result()
            stage = "download"
            shard = fut_mdl.result()

        stage = "server"
        write_status("running", "server", f"Starting llama-server ({LLAMA_PORT}) and Open WebUI ({OW_PORT})...")
        llama_api_key, llama_proc = start_llama_server(binary=binary, shard=shard)
        llama_api_key_box.append(llama_api_key)
        access, ow_proc = start_openwebui(llama_api_key=llama_api_key)

        stage = "verify"
        verify_services(
            llama_api_key=llama_api_key,
            access=access,
            llama_proc=llama_proc,
            ow_proc=ow_proc,
        )

        public_url = start_cloudflare_tunnel() if ENABLE_CF_TUNNEL else None
        gpu = query_gpu_summary()
        outputs: dict[str, Any] = {
            "model_id": LLAMA_MODEL,
            "login_email": access["email"],
            "login_password": access["password"],
            "ctx_size": CTX_SIZE,
            "openwebui_version": OPENWEBUI_VERSION,
            "gpu": gpu.get("name", "unknown"),
            "vram_used_mib": int(gpu.get("vram_used_mb", 0)),
            "vram_total_mib": int(gpu.get("vram_total_mb", 0)),
        }
        if public_url:
            outputs["cloudflare_url"] = public_url

        setup_done.set()
        write_status(
            "ready",
            "verify",
            f"Open WebUI ready ({access['email']}) · {LLAMA_MODEL} on {gpu.get('name', 'GPU')}",
            progress=1.0,
            outputs=outputs,
        )
        log_step("Ready! Monitoring llama-server and Open WebUI...")

        # Keep alive while monitoring child processes.
        while not _stop_event.wait(10.0):
            if llama_proc is not None and llama_proc.poll() is not None:
                fail("server", f"llama-server exited unexpectedly (code {llama_proc.returncode})")
            if ow_proc is not None and ow_proc.poll() is not None:
                fail("server", f"Open WebUI exited unexpectedly (code {ow_proc.returncode})")
    except SystemExit:
        raise
    except Exception as exc:
        fail(stage, str(exc))


if __name__ == "__main__":
    main()
