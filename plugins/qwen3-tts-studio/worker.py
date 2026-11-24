#!/usr/bin/env python3
"""Qwen3-TTS 1.7B Gradio WebUI worker for Colab Playground."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

E = os.environ.get
RUN_DIR = Path(E("PLUGIN_RUN_DIR", "/root/.colab-playground/runs/manual"))
STATUS_FILE = Path(E("PLUGIN_STATUS_FILE", str(RUN_DIR / "status.json")))
HEARTBEAT = E("PLUGIN_HEARTBEAT", "")

MODE = (E("PLUGIN_INPUT_MODE", "CustomVoice-1.7B") or "CustomVoice-1.7B").strip()
PORT = int(float(E("PLUGIN_INPUT_PORT", "7860") or 7860))
HF_TOKEN = (E("PLUGIN_HF_TOKEN", "") or "").strip()

MODE_MODELS: dict[str, str] = {
    "CustomVoice-1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "VoiceClone-1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "VoiceDesign-1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}

_status_lock = threading.Lock()
_stop_event = threading.Event()
_child_procs: list[subprocess.Popen] = []


def log_step(msg: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[qwen3-tts {stamp}] {msg}", flush=True)


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


def _heartbeat_loop(setup_done: threading.Event) -> None:
    while not _stop_event.wait(30.0):
        if not setup_done.is_set():
            touch_heartbeat()


def query_gpu() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"name": "CPU", "vram_total_mb": 0.0}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip()
        parts = [p.strip() for p in out.splitlines()[0].split(",")]
        return {"name": parts[0], "vram_total_mb": float(parts[1])}
    except Exception:
        return {"name": "CPU", "vram_total_mb": 0.0}


def _http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return 200 <= r.status < 400
    except Exception:
        return False


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _on_sigterm)
    signal.signal(signal.SIGINT, _on_sigterm)

    if PORT == 8080:
        fail("install", "VM port 8080 is reserved by Colab; choose another port.")

    model_id = MODE_MODELS.get(MODE)
    if not model_id:
        fail("install", f"Unknown mode: {MODE}")

    setup_done = threading.Event()
    threading.Thread(target=_heartbeat_loop, args=(setup_done,), daemon=True).start()
    touch_heartbeat()

    stage = "install"
    try:
        write_status("running", "install", "Installing qwen-tts & hf_transfer...")
        if not shutil.which("qwen-tts-demo"):
            log_step("Installing qwen-tts package...")
            uv = shutil.which("uv")
            if uv:
                subprocess.check_call([uv, "pip", "install", "--system", "-q", "-U", "qwen-tts", "hf_transfer"])
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "qwen-tts", "hf_transfer"])
        else:
            log_step("qwen-tts-demo already installed.")

        stage = "server"
        write_status("running", "server", f"Downloading & loading {model_id} on port {PORT}...")
        subprocess.run(["pkill", "-f", f"qwen-tts-demo.*--port {PORT}"], check=False)
        time.sleep(1.0)

        env = dict(os.environ, HF_HUB_ENABLE_HF_TRANSFER="1")
        if HF_TOKEN:
            env["HF_TOKEN"] = HF_TOKEN

        demo_bin = shutil.which("qwen-tts-demo") or "qwen-tts-demo"
        server_log = RUN_DIR / "qwen3-tts.log"
        cmd = [demo_bin, model_id, "--ip", "127.0.0.1", "--port", str(PORT), "--no-flash-attn"]
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=open(server_log, "w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _child_procs.append(proc)
        log_step(f"Started qwen-tts-demo (PID {proc.pid}) for {model_id} on 127.0.0.1:{PORT}.")

        for _ in range(300):
            if proc.poll() is not None:
                tail = server_log.read_text(errors="replace")[-1500:]
                raise RuntimeError(f"qwen-tts-demo exited early ({proc.returncode}): {tail}")
            if _http_ok(f"http://127.0.0.1:{PORT}/"):
                break
            time.sleep(2.0)
        else:
            raise TimeoutError(f"Timed out waiting for Qwen3-TTS WebUI on port {PORT}")

        gpu = query_gpu()
        outputs = {
            "mode": MODE,
            "model_id": model_id,
            "gpu": gpu["name"],
            "vram_total_mib": int(gpu["vram_total_mb"]),
        }
        setup_done.set()
        write_status("ready", "server", f"Qwen3-TTS ({MODE}) ready on {gpu['name']}", progress=1.0, outputs=outputs)
        log_step(f"Ready! Qwen3-TTS ({MODE}) serving on 127.0.0.1:{PORT}.")

        while not _stop_event.wait(10.0):
            if proc.poll() is not None:
                fail("server", f"qwen-tts-demo exited unexpectedly ({proc.returncode})")
    except SystemExit:
        raise
    except Exception as exc:
        fail(stage, str(exc))


if __name__ == "__main__":
    main()
