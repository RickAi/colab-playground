#!/usr/bin/env python3
"""ComfyUI (Qwen-Image-2.1 BF16 + MiniMax H3 + AI Pose Gallery + 9 workflows) worker for Colab Playground.

Adapted from Colab Studio's `colab_worker.py` & `qwen_h3_bundle.py`
(`Qwen-Image-to-H3-ComfyUI-本地磁盘.ipynb`) for Colab Playground's plugin protocol (`docs/PLUGINS.md`).
"""

from __future__ import annotations

import ast
import base64
import copy
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
import zipfile
import zlib
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote, urlparse

E = os.environ.get
RUN_DIR = Path(E("PLUGIN_RUN_DIR", "/root/.colab-playground/runs/manual"))
STATUS_FILE = Path(E("PLUGIN_STATUS_FILE", str(RUN_DIR / "status.json")))
HEARTBEAT = E("PLUGIN_HEARTBEAT", "")

PRESET = (E("PLUGIN_INPUT_PRESET", "qwen_h3") or "qwen_h3").strip()
COMFY_PORT = int(float(E("PLUGIN_INPUT_PORT", "8188") or 8188))
BG_MODEL_SYNC = E("PLUGIN_INPUT_BACKGROUND_MODEL_SYNC", "1") == "1"
ENABLE_CF_TUNNEL = E("PLUGIN_INPUT_CLOUDFLARE_TUNNEL", "0") == "1"
HF_TOKEN = (E("PLUGIN_HF_TOKEN", "") or "").strip()

COMFY_DIR = Path("/content/ComfyUI")
H3_WORKSPACE = Path("/content/MiniMax_H3_Workspace")
CLOUDFLARED_BIN = Path("/content/cloudflared")

PRIMARY_QWEN_H3_WORKFLOW = "Qwen-Pose-First-Last-Frame-to-H3.json"
TAB_QWEN_H3_WORKFLOWS = [
    "Qwen-Pose-First-Last-Frame-to-H3.json",
    "Qwen-Edit-First-Last-Frame-to-H3.json",
    "Qwen-Native-Image-Edit-to-H3-Adaptive.json",
    "Qwen-Pose-to-Last-Frame.json",
    "Qwen-Pose-First-Last-Frame-to-H3-Telegram.json",
    "minimax_h3_t2v.json",
]
ALL_QWEN_H3_WORKFLOWS = [
    "Qwen-Pose-First-Last-Frame-to-H3.json",
    "Qwen-Edit-First-Last-Frame-to-H3.json",
    "Qwen-Native-Image-Edit-to-H3-Adaptive.json",
    "Qwen-Pose-to-Last-Frame.json",
    "Qwen-Pose-First-Last-Frame-to-H3-Telegram.json",
    "Qwen-Edit-First-Last-Frame-to-H3-Telegram.json",
    "Qwen-Native-Image-Edit-to-H3-Adaptive-Telegram.json",
    "Qwen-Image-to-H3-Video.json",
    "minimax_h3_t2v.json",
]

_status_lock = threading.Lock()
_stop_event = threading.Event()
_child_procs: list[subprocess.Popen] = []
_sync_busy = threading.Event()


def log_step(msg: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[comfyui-qwen-h3 {stamp}] {msg}", flush=True)


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
        if not setup_done.is_set() or _sync_busy.is_set():
            touch_heartbeat()
            continue
        # Once ready, touch heartbeat whenever ComfyUI queue has running or pending jobs.
        try:
            import requests

            r = requests.get(f"http://127.0.0.1:{COMFY_PORT}/queue", timeout=4)
            if r.ok:
                data = r.json()
                if data.get("queue_running") or data.get("queue_pending"):
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


def _install_pip_packages(args: list[str]) -> None:
    if shutil.which("uv"):
        subprocess.check_call(["uv", "pip", "install", "--system", "-q", *args])
    else:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])


def _extract_assigned_str(source: str, var_name: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == var_name:
                    val = ast.literal_eval(node.value)
                    if isinstance(val, str):
                        return val
    raise RuntimeError(f"Variable {var_name} not found in notebook cell")


def _hf_download_fast(repo_id: str, filename: str, revision: str, local_dir: str) -> Path:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    os.environ["HF_HOME"] = "/content/huggingface_cache"
    token_arg: Any = HF_TOKEN if HF_TOKEN else None
    try:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        p = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_dir=local_dir,
                token=token_arg,
            )
        )
    except Exception:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        p = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_dir=local_dir,
                token=token_arg,
            )
        )
    shutil.rmtree(Path(local_dir) / ".cache", ignore_errors=True)
    return p


def _upgrade_workflow_with_telegram(original: dict, workflow_stem: str = "") -> Optional[dict]:
    w = copy.deepcopy(original)
    nodes = w.get("nodes", [])
    by_id = {n["id"]: n for n in nodes}
    uploads = [n for n in nodes if n.get("type") == "TelegramInputVideo"]
    images = [n for n in nodes if n.get("type") == "LoadImage"]
    videos = [n for n in nodes if n.get("type") == "SaveVideo"]
    if len(images) != 1 or len(videos) != 1:
        return None
    vi = videos[0]
    vl = next(l for l in w["links"] if l[0] == vi["inputs"][0]["link"])
    sources = [("image", "IMAGE", images[0]["id"], 0), ("video", "VIDEO", vl[1], vl[2])]
    editors = [n for n in nodes if n.get("type") == "TextEncodeQwenImage21"]

    def source_for(node: dict, name: str):
        inp = next((i for i in node.get("inputs", []) if i.get("name") == name), None)
        if not inp or inp.get("link") is None:
            return None
        link = next(l for l in w["links"] if l[0] == inp["link"])
        return link[1], link[2]

    if editors:
        for name, typ, source in [
            ("pose", "IMAGE", source_for(editors[0], "images.image_2")),
            ("edit_prompt", "STRING", source_for(editors[0], "prompt")),
            ("video_prompt", "STRING", source_for(by_id[vl[1]], "prompt")),
        ]:
            if source:
                sources.append((name, typ, *source))

    generated_saves = [
        n
        for n in nodes
        if n.get("type") == "SaveImage"
        and (n.get("widgets_values") or [""])[0] == "Pose_first_last/last_frame"
    ]
    generated = (
        source_for(generated_saves[0], "images")
        if len(generated_saves) == 1
        else source_for(by_id[vl[1]], "last_frame")
    )
    if generated:
        sources.append(("generated_image", "IMAGE", *generated))
    if any(n.get("type") == "PoseGalleryNode" for n in nodes):
        assert len(sources) == 6, "Pose workflow must expose both prompts, pose and generated image"

    if uploads:
        assert len(uploads) == 1
        node = uploads[0]
        old_links = {i["link"] for i in node.get("inputs", []) if i.get("link") is not None}
        w["links"] = [l for l in w["links"] if l[0] not in old_links]
        for n in nodes:
            for output in n.get("outputs", []):
                if output.get("links"):
                    output["links"] = [i for i in output["links"] if i not in old_links]
    else:
        node = {
            "id": max(by_id) + 1,
            "type": "TelegramInputVideo",
            "pos": [vi["pos"][0] + 700, vi["pos"][1]],
            "size": [360, 190],
            "flags": {},
            "order": len(nodes),
            "mode": 0,
            "outputs": [],
            "widgets_values": [True],
            "properties": {"Node name for S&R": "TelegramInputVideo"},
        }
        nodes.append(node)
        seed_key = workflow_stem or str(w.get("id") or "")
        w["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"colab-yoga-telegram:{seed_key}"))

    node["inputs"] = []
    node.setdefault("properties", {})["telegram_upload_version"] = 3
    for slot, (name, typ, src, out) in enumerate(sources):
        lid = max([l[0] for l in w["links"]] + [0]) + 1
        w["links"].append([lid, src, out, node["id"], slot, typ])
        output = by_id[src]["outputs"][out]
        output["links"] = (output.get("links") or []) + [lid]
        node["inputs"].append({"name": name, "type": typ, "link": lid})

    w["last_node_id"] = max(n["id"] for n in nodes)
    w["last_link_id"] = max(l[0] for l in w["links"])
    return w


COLAB_YOGA_AUTOLOAD_INIT_PY = '''"""Default workflow endpoint & frontend auto-loader for ComfyUI."""
from pathlib import Path

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"

PRIMARY_WORKFLOW = "Qwen-Pose-First-Last-Frame-to-H3.json"
TAB_WORKFLOWS = [
    "Qwen-Pose-First-Last-Frame-to-H3.json",
    "Qwen-Edit-First-Last-Frame-to-H3.json",
    "Qwen-Native-Image-Edit-to-H3-Adaptive.json",
    "Qwen-Pose-to-Last-Frame.json",
    "Qwen-Pose-First-Last-Frame-to-H3-Telegram.json",
    "minimax_h3_t2v.json",
]
ALL_NOTEBOOK_WORKFLOWS = [
    "Qwen-Pose-First-Last-Frame-to-H3.json",
    "Qwen-Edit-First-Last-Frame-to-H3.json",
    "Qwen-Native-Image-Edit-to-H3-Adaptive.json",
    "Qwen-Pose-to-Last-Frame.json",
    "Qwen-Pose-First-Last-Frame-to-H3-Telegram.json",
    "Qwen-Edit-First-Last-Frame-to-H3-Telegram.json",
    "Qwen-Native-Image-Edit-to-H3-Adaptive-Telegram.json",
    "Qwen-Image-to-H3-Video.json",
    "minimax_h3_t2v.json",
]


def _workflows_dir() -> Path:
    try:
        import folder_paths
        user_dir = Path(folder_paths.get_user_directory())
        cand = user_dir / "default" / "workflows"
        if cand.is_dir():
            return cand
    except Exception:
        pass
    return Path(__file__).resolve().parents[2] / "user" / "default" / "workflows"


try:
    from aiohttp import web
    from server import PromptServer

    if getattr(PromptServer, "instance", None) is not None:
        routes = PromptServer.instance.routes
        has_default = any(
            getattr(r, "path", None) == "/default_workflow" and getattr(r, "method", None) == "GET"
            for r in routes
        )
        if not has_default:
            @routes.get("/default_workflow")
            async def get_default_workflow(request):
                wf_dir = _workflows_dir()
                target = wf_dir / PRIMARY_WORKFLOW
                if not target.is_file():
                    files = sorted(wf_dir.glob("*.json"))
                    if not files:
                        return web.json_response({"error": "No default workflow found"}, status=404)
                    target = files[0]
                return web.Response(
                    text=target.read_text(encoding="utf-8"),
                    content_type="application/json",
                )

        has_yoga_wf = any(
            getattr(r, "path", None) == "/colab_yoga/workflows" and getattr(r, "method", None) == "GET"
            for r in routes
        )
        if not has_yoga_wf:
            @routes.get("/colab_yoga/workflows")
            async def get_colab_yoga_workflows(request):
                wf_dir = _workflows_dir()
                available = {p.name for p in wf_dir.glob("*.json") if p.is_file()}
                ordered_all = [w for w in ALL_NOTEBOOK_WORKFLOWS if w in available]
                for extra in sorted(available):
                    if extra not in ordered_all:
                        ordered_all.append(extra)
                primary = (
                    PRIMARY_WORKFLOW
                    if PRIMARY_WORKFLOW in available
                    else (ordered_all[0] if ordered_all else None)
                )
                tabs = [w for w in TAB_WORKFLOWS if w in available]
                return web.json_response({
                    "primary": f"workflows/{primary}" if primary else None,
                    "tabs": [f"workflows/{w}" for w in tabs],
                    "all": [f"workflows/{w}" for w in ordered_all],
                })
except Exception as exc:
    print(f"[colab_yoga_workflows] route registration note: {exc}", flush=True)
'''

COLAB_YOGA_AUTOLOAD_JS = '''import { app } from "../../scripts/app.js";

const FALLBACK_PRIMARY = "workflows/Qwen-Pose-First-Last-Frame-to-H3.json";
const FALLBACK_TABS = [
  "workflows/Qwen-Pose-First-Last-Frame-to-H3.json",
  "workflows/Qwen-Edit-First-Last-Frame-to-H3.json",
  "workflows/Qwen-Native-Image-Edit-to-H3-Adaptive.json",
  "workflows/Qwen-Pose-to-Last-Frame.json",
  "workflows/Qwen-Pose-First-Last-Frame-to-H3-Telegram.json",
  "workflows/minimax_h3_t2v.json",
];

let _autoloadExecuted = false;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function refreshPosePreviews() {
  const nodes = app.rootGraph?._nodes || [];
  for (const node of nodes) {
    if (node?.type === "PoseGalleryNode") {
      node.pcaRefreshPreview?.();
    }
  }
}

async function _runWorkflowAutoload() {
  let primaryPath = FALLBACK_PRIMARY;
  let tabPaths = [...FALLBACK_TABS];
  try {
    const metaUrl = new URL("api/colab_yoga/workflows", window.location.href);
    const resp = await fetch(metaUrl.toString(), { cache: "no-store" });
    if (resp.ok) {
      const meta = await resp.json();
      if (meta?.primary) primaryPath = meta.primary;
      if (Array.isArray(meta?.tabs) && meta.tabs.length > 0) tabPaths = meta.tabs;
    }
  } catch (_) {}

  let wfStore = null;
  for (let i = 0; i < 60; i++) {
    wfStore = app.extensionManager?.workflow;
    if (app.rootGraph && wfStore && (wfStore.tabsRestored === true || (i >= 12 && wfStore.activeWorkflow != null))) {
      break;
    }
    await sleep(100);
  }
  await sleep(80);

  wfStore = app.extensionManager?.workflow;
  if (!app.rootGraph) return;

  if (wfStore && typeof wfStore.loadWorkflows === "function") {
    try {
      await wfStore.loadWorkflows();
    } catch (_) {}
  }

  const activeWf = wfStore?.activeWorkflow || null;
  const nodes = Array.isArray(app.rootGraph?._nodes) ? app.rootGraph._nodes : [];
  const isEmptyCanvas = nodes.length === 0;
  const isBuiltinSd15Default =
    nodes.length <= 8 &&
    nodes.some(
      (n) =>
        n?.type === "CheckpointLoaderSimple" &&
        JSON.stringify(n?.widgets_values || []).includes("v1-5-pruned-emaonly")
    );
  const isUnboundTemp = !activeWf || Boolean(activeWf.isTemporary);
  const hasCustomUserWork =
    !isEmptyCanvas &&
    !isBuiltinSd15Default &&
    Boolean(activeWf?.isTemporary && activeWf?.isModified) &&
    !nodes.some((n) => n?.type === "PoseGalleryNode" || n?.type === "PoseLastFramePrompt");

  const needsPrimaryLoad = isEmptyCanvas || isBuiltinSd15Default || (isUnboundTemp && !hasCustomUserWork);

  if (needsPrimaryLoad) {
    const tempBefore = activeWf && activeWf.isTemporary ? activeWf : null;
    const primaryWf = wfStore?.getWorkflowByPath?.(primaryPath) || null;
    let loadedOk = false;

    if (primaryWf) {
      try {
        if (!primaryWf.isLoaded && typeof primaryWf.load === "function") {
          await primaryWf.load();
        }
        const rawState = primaryWf.activeState || (primaryWf.content ? JSON.parse(primaryWf.content) : null);
        if (rawState) {
          const graphClone = JSON.parse(JSON.stringify(rawState));
          await app.loadGraphData(graphClone, true, true, primaryWf, {
            checkForRerouteMigration: false,
            deferWarnings: true,
            skipAssetScans: true,
            silentAssetErrors: true,
          });
          loadedOk = true;
        }
        if (
          loadedOk &&
          tempBefore &&
          tempBefore !== primaryWf &&
          wfStore?.isOpen?.(tempBefore) &&
          typeof wfStore.closeWorkflow === "function"
        ) {
          try {
            await wfStore.closeWorkflow(tempBefore);
          } catch (_) {}
        }
      } catch (err) {
        console.warn("[ColabYoga] Primary workflow store open fallback:", err);
      }
    }

    if (!loadedOk) {
      try {
        const defUrl = new URL("api/default_workflow", window.location.href);
        const resp = await fetch(defUrl.toString(), { cache: "no-store" });
        if (resp.ok) {
          const graphObj = await resp.json();
          await app.loadGraphData(graphObj, true, true, primaryPath.replace(/^workflows\\//, ""), {
            checkForRerouteMigration: false,
            deferWarnings: true,
            skipAssetScans: true,
            silentAssetErrors: true,
          });
        }
      } catch (err) {
        console.warn("[ColabYoga] Fallback /api/default_workflow failed:", err);
      }
    }
  }

  if (wfStore && typeof wfStore.openWorkflowsInBackground === "function") {
    try {
      const activePath = wfStore.activeWorkflow?.path || primaryPath;
      const bgPaths = tabPaths.filter((p) => {
        if (p === activePath) return false;
        const w = wfStore.getWorkflowByPath?.(p);
        if (!w) return false;
        const alreadyOpen = wfStore.isOpen ? wfStore.isOpen(w) : (wfStore.openWorkflows || []).includes(w);
        return !alreadyOpen;
      });
      if (bgPaths.length > 0) {
        wfStore.openWorkflowsInBackground({ right: bgPaths });
      }
    } catch (_) {}
  }

  setTimeout(refreshPosePreviews, 250);
}

app.registerExtension({
  name: "ColabYoga.WorkflowAutoload",
  setup() {
    if (_autoloadExecuted) return;
    _autoloadExecuted = true;
    try {
      app.extensionManager?.setting?.set?.("Comfy.TutorialCompleted", true);
    } catch (_) {}
    setTimeout(() => {
      void _runWorkflowAutoload();
    }, 30);
  },
});
'''


def _install_colab_yoga_autoload_node(comfy_root: Path) -> None:
    ext_root = comfy_root / "custom_nodes" / "colab_yoga_workflows"
    web_dir = ext_root / "web"
    web_dir.mkdir(parents=True, exist_ok=True)
    (ext_root / "__init__.py").write_text(COLAB_YOGA_AUTOLOAD_INIT_PY, encoding="utf-8")
    (web_dir / "autoload_workflows.js").write_text(COLAB_YOGA_AUTOLOAD_JS, encoding="utf-8")


def _install_qwen_h3_workspace(
    comfy_root: Path,
    ws: Path,
    cells: dict[str, Any],
) -> tuple[dict[str, Any], Any, list[str]]:
    os.environ["HF_HOME"] = "/content/huggingface_cache"
    for sub in ("models", "outputs", "workflows"):
        (ws / sub).mkdir(parents=True, exist_ok=True)
    for msub in ("diffusion_models", "vae", "text_encoders", "loras"):
        (ws / "models" / msub).mkdir(parents=True, exist_ok=True)
    (comfy_root / "input").mkdir(parents=True, exist_ok=True)
    (comfy_root / "custom_nodes").mkdir(parents=True, exist_ok=True)

    (comfy_root / "extra_model_paths.yaml").write_text(
        f"minimax_h3:\n"
        f"  base_path: {ws / 'models'}\n"
        f"  diffusion_models: diffusion_models\n"
        f"  vae: vae\n"
        f"  text_encoders: text_encoders\n"
        f"  loras: loras\n",
        encoding="utf-8",
    )

    user_default_dir = comfy_root / "user" / "default"
    wf_dirs = [ws / "workflows", user_default_dir / "workflows"]
    for d in wf_dirs:
        d.mkdir(parents=True, exist_ok=True)

    # 1. MiniMax H3 T2V template (live fetch with bundled offline fallback)
    tpl_workflow = None
    tpl_url = "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_minimax_h3_t2v.json"
    try:
        with urllib.request.urlopen(tpl_url, timeout=45) as resp:
            tpl_workflow = json.load(resp)
    except Exception:
        tpl_workflow = None
    if tpl_workflow is None:
        if "minimax_h3_t2v_fallback" not in cells:
            raise RuntimeError("MiniMax H3 template missing and network fetch failed")
        tpl_workflow = json.loads(cells["minimax_h3_t2v_fallback"])

    tpl_str = json.dumps(tpl_workflow, ensure_ascii=False, indent=2)
    for d in wf_dirs:
        (d / "minimax_h3_t2v.json").write_text(tpl_str, encoding="utf-8")

    # 2. QwenImage21Turbo custom node + Qwen-Image-to-H3-Video.json
    if "9" in cells and "10" in cells:
        qwen_turbo_src = _extract_assigned_str(cells["9"], "NODE_SOURCE")
        qwen_turbo_dir = comfy_root / "custom_nodes" / "qwen_image_21_turbo"
        qwen_turbo_dir.mkdir(parents=True, exist_ok=True)
        compile(qwen_turbo_src, str(qwen_turbo_dir / "__init__.py"), "exec")
        (qwen_turbo_dir / "__init__.py").write_text(qwen_turbo_src, encoding="utf-8")

        wf10 = _extract_assigned_str(cells["10"], "WORKFLOW_JSON")
        for d in wf_dirs:
            (d / "Qwen-Image-to-H3-Video.json").write_text(wf10, encoding="utf-8")

    # 3. Native BF16 edit & First-Last-Frame workflows
    wf11 = _extract_assigned_str(cells["11"], "NATIVE_EDIT_WORKFLOW_JSON")
    wf12 = _extract_assigned_str(cells["12"], "workflow_json")
    for d in wf_dirs:
        (d / "Qwen-Native-Image-Edit-to-H3-Adaptive.json").write_text(wf11, encoding="utf-8")
        (d / "Qwen-Edit-First-Last-Frame-to-H3.json").write_text(wf12, encoding="utf-8")

    # 4. AI Pose Gallery custom node + 2 Pose workflows
    m13 = re.search(r"b64decode\((['\"])(.+?)\1\)", cells["13"])
    if not m13:
        raise RuntimeError("Pose gallery bundle missing in Cell 13")
    payload = base64.b64decode(m13.group(2))
    if hashlib.sha256(payload).hexdigest() != "edfc8fffd70cdb59b2a396579d200f74ddf9a36f54d0af4e436a66b62af545d8":
        raise RuntimeError("Pose gallery bundle SHA-256 verification failed")
    bundle_dir = ws / "pose_workflow_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(bundle_dir)

    spec = importlib.util.spec_from_file_location("pose_workflow_installer", bundle_dir / "install.py")
    assert spec is not None and spec.loader is not None
    pose_installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pose_installer)
    pose_installer.install(comfy_root, ws, bundle_dir)

    # 5. TelegramInputVideo custom node + 3 *-Telegram.json upgraded workflows
    if "15" in cells:
        tg_node_src = _extract_assigned_str(cells["15"], "node_source")
        tg_dir = comfy_root / "custom_nodes" / "telegram_input_video"
        tg_dir.mkdir(parents=True, exist_ok=True)
        compile(tg_node_src, str(tg_dir / "__init__.py"), "exec")
        (tg_dir / "__init__.py").write_text(tg_node_src, encoding="utf-8")

        user_wf_dir = user_default_dir / "workflows"
        for src_path in sorted(user_wf_dir.glob("*.json")):
            if src_path.stem.endswith("-Telegram"):
                continue
            target_name = f"{src_path.stem}-Telegram.json"
            existing_tg = user_wf_dir / target_name
            base_obj = json.loads((existing_tg if existing_tg.is_file() else src_path).read_text(encoding="utf-8"))
            upgraded = _upgrade_workflow_with_telegram(base_obj, workflow_stem=src_path.stem)
            if upgraded is None:
                continue
            upgraded_text = json.dumps(upgraded, ensure_ascii=False)
            for d in wf_dirs:
                (d / target_name).write_text(upgraded_text, encoding="utf-8")

    # 6. Mark Comfy.TutorialCompleted=True so ComfyUI_frontend doesn't blank the canvas
    settings_file = user_default_dir / "comfy.settings.json"
    existing_settings: dict[str, Any] = {}
    if settings_file.is_file():
        try:
            loaded = json.loads(settings_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing_settings = loaded
        except Exception:
            existing_settings = {}
    existing_settings["Comfy.TutorialCompleted"] = True
    settings_file.write_text(json.dumps(existing_settings, ensure_ascii=False, indent=2), encoding="utf-8")

    # 7. Install frontend canvas auto-loader extension
    _install_colab_yoga_autoload_node(comfy_root)

    installed_names = [
        name for name in ALL_QWEN_H3_WORKFLOWS if (user_default_dir / "workflows" / name).is_file()
    ]
    return tpl_workflow, pose_installer, installed_names


def _sync_qwen_h3_models(tpl_workflow: dict[str, Any], update_run_status: bool) -> None:
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
    import requests

    _sync_busy.set()
    try:
        ws = H3_WORKSPACE
        model_root = ws / "models"
        log_step("Resolving HuggingFace model metadata for MiniMax H3 + Qwen-Image-2.1 BF16...")

        h3_models: dict[str, Any] = {}

        def collect_models(value: Any) -> None:
            if isinstance(value, dict):
                for m in value.get("models", []):
                    h3_models[m["url"]] = m
                for child in value.values():
                    collect_models(child)
            elif isinstance(value, list):
                for child in value:
                    collect_models(child)

        collect_models(tpl_workflow)

        tasks: list[dict[str, Any]] = []
        for url, model in h3_models.items():
            parsed = urlparse(url)
            parts = parsed.path.strip("/").split("/")
            if not (parsed.scheme == "https" and parsed.netloc == "huggingface.co" and parts[2] == "resolve"):
                continue
            repo_id = "/".join(parts[:2])
            revision = parts[3]
            filename = unquote("/".join(parts[4:]))
            target_dir = model_root / model["directory"]
            target_path = target_dir / model["name"]
            meta = get_hf_file_metadata(hf_hub_url(repo_id, filename, revision=revision))
            tasks.append({
                "label": model["name"],
                "repo_id": repo_id,
                "revision": revision,
                "filename": filename,
                "local_dir": str(target_dir),
                "target_path": target_path,
                "expected_size": meta.size or 0,
            })

        qwen_repo = "Comfy-Org/Qwen-Image-2.1"
        qwen_rev_pin = "a8342359c61979b06a8dafb1a47991fe1f30e994"
        qwen_files = [
            "diffusion_models/qwen_image_2.1_bf16.safetensors",
            "text_encoders/qwen3vl_8b_bf16.safetensors",
            "vae/qwen_image_2.1_vae_bf16.safetensors",
        ]
        info = HfApi(token=HF_TOKEN or None).model_info(qwen_repo, revision=qwen_rev_pin, files_metadata=True)
        qwen_sha = info.sha
        qwen_sizes = {f.rfilename: (f.size or 0) for f in info.siblings}
        for rel_file in qwen_files:
            tasks.append({
                "label": os.path.basename(rel_file),
                "repo_id": qwen_repo,
                "revision": qwen_sha,
                "filename": rel_file,
                "local_dir": str(model_root),
                "target_path": model_root / rel_file,
                "expected_size": qwen_sizes.get(rel_file, 0),
            })

        total = len(tasks)
        for idx, item in enumerate(tasks, start=1):
            target: Path = item["target_path"]
            exp_sz: int = item["expected_size"]
            gib = exp_sz / (1024 ** 3) if exp_sz else 0.0
            label = item["label"]

            if target.is_file() and (exp_sz <= 0 or target.stat().st_size == exp_sz):
                log_step(f"[{idx}/{total}] {label} already present.")
                continue

            free_bytes = shutil.disk_usage(str(ws)).free
            if exp_sz > 0 and free_bytes < exp_sz + 512 * 1024 * 1024:
                raise RuntimeError(
                    f"Insufficient disk space ({free_bytes / (1024**3):.1f} GiB free) for {label} ({gib:.1f} GiB). Enable High-RAM."
                )

            msg = f"[{idx}/{total}] Downloading {label} ({gib:.1f} GiB)..."
            log_step(msg)
            if update_run_status:
                write_status("running", "models", msg, progress=(idx - 1) / max(1, total))

            target.parent.mkdir(parents=True, exist_ok=True)
            downloaded = _hf_download_fast(
                repo_id=item["repo_id"],
                filename=item["filename"],
                revision=item["revision"],
                local_dir=item["local_dir"],
            )
            if downloaded != target:
                downloaded.replace(target)
            if exp_sz > 0 and target.stat().st_size != exp_sz:
                raise RuntimeError(f"Size mismatch for {label}: got {target.stat().st_size}, expected {exp_sz}")

        (ws / "workflows" / "qwen-native-models.json").write_text(
            json.dumps({"repo": qwen_repo, "revision": qwen_sha, "files": qwen_files}, indent=2),
            encoding="utf-8",
        )
        try:
            requests.get(f"http://127.0.0.1:{COMFY_PORT}/object_info", timeout=10)
        except Exception:
            pass
        log_step(f"All {total}/{total} Qwen-Image-2.1 BF16 + MiniMax H3 models are ready!")
        if update_run_status:
            write_status("running", "models", f"All {total}/{total} models synced", progress=1.0)
    finally:
        _sync_busy.clear()


def _start_cloudflare_tunnel() -> Optional[str]:
    if not CLOUDFLARED_BIN.is_file():
        log_step("Downloading cloudflared...")
        urllib.request.urlretrieve(
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            CLOUDFLARED_BIN,
        )
        CLOUDFLARED_BIN.chmod(0o700)

    tlog_path = RUN_DIR / "cloudflared.log"
    tlog_f = open(tlog_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            str(CLOUDFLARED_BIN),
            "tunnel",
            "--no-autoupdate",
            "--protocol",
            "http2",
            "--url",
            f"http://127.0.0.1:{COMFY_PORT}",
        ],
        stdin=subprocess.DEVNULL,
        stdout=tlog_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _child_procs.append(proc)
    pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    for _ in range(45):
        if tlog_path.is_file():
            for m in pattern.findall(tlog_path.read_text(errors="ignore")):
                if "api.trycloudflare.com" not in m:
                    log_step(f"Cloudflare public URL ready: {m}")
                    return m
        if proc.poll() is not None:
            return None
        time.sleep(1.0)
    return None


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _on_sigterm)
    signal.signal(signal.SIGINT, _on_sigterm)

    if COMFY_PORT == 8080:
        fail("install", "VM port 8080 is reserved by Colab; choose another port.")

    setup_done = threading.Event()
    threading.Thread(target=_heartbeat_loop, args=(setup_done,), daemon=True).start()
    touch_heartbeat()

    stage = "install"
    try:
        write_status("running", "install", "Cloning Comfy-Org/ComfyUI and installing dependencies...")
        if not (COMFY_DIR / "main.py").is_file():
            log_step("Cloning official Comfy-Org/ComfyUI repository...")
            subprocess.check_call(
                ["git", "clone", "--depth", "1", "https://github.com/Comfy-Org/ComfyUI.git", str(COMFY_DIR)]
            )
            log_step("Installing ComfyUI requirements & huggingface_hub/hf_transfer...")
            _install_pip_packages(
                ["-r", str(COMFY_DIR / "requirements.txt"), "huggingface_hub", "hf_transfer", "requests"]
            )
        else:
            log_step("ComfyUI already cloned; ensuring helper packages...")
            _install_pip_packages(["huggingface_hub", "hf_transfer", "requests"])

        import requests

        user_dir = COMFY_DIR / "user"
        (user_dir / "default" / "workflows").mkdir(parents=True, exist_ok=True)

        comfy_cmd = [
            sys.executable,
            "main.py",
            "--listen",
            "127.0.0.1",
            "--port",
            str(COMFY_PORT),
            "--user-directory",
            str(user_dir),
            "--preview-method",
            "auto",
            "--enable-cors-header",
        ]

        stage = "workspace"
        tpl_workflow = None
        pose_installer = None
        installed_workflows: list[str] = []

        if PRESET == "qwen_h3":
            write_status("running", "workspace", "Installing Qwen-Image-2.1, MiniMax H3, AI Pose Gallery & 9 workflows...")
            log_step("Unpacking embedded Qwen-Image-to-H3 workflow bundle...")
            cells = json.loads(zlib.decompress(base64.b64decode(QWEN_H3_CELLS_B64)).decode("utf-8"))
            tpl_workflow, pose_installer, installed_workflows = _install_qwen_h3_workspace(
                comfy_root=COMFY_DIR,
                ws=H3_WORKSPACE,
                cells=cells,
            )
            comfy_cmd.extend(["--output-directory", str(H3_WORKSPACE / "outputs")])
            log_step(f"Installed {len(installed_workflows)} workflows into ComfyUI.")
        else:
            write_status("running", "workspace", "Preparing SDXL-Turbo FP16 starter checkpoint...")
            ckpt_dir = COMFY_DIR / "models" / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            existing_ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith((".safetensors", ".ckpt", ".pt"))]
            if not existing_ckpts:
                log_step("Downloading SDXL-Turbo FP16 (~6.9 GB)...")
                ckpt_url = "https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors"
                target_ckpt = ckpt_dir / "sd_xl_turbo_1.0_fp16.safetensors"
                subprocess.check_call(["wget", "-q", "-O", str(target_ckpt), ckpt_url])

        stage = "server"
        write_status("running", "server", f"Starting ComfyUI on port {COMFY_PORT} and verifying custom nodes...")
        subprocess.run(["pkill", "-f", f"main.py.*--port {COMFY_PORT}"], check=False)
        time.sleep(1.0)

        comfy_log = RUN_DIR / "comfyui.log"
        log_f = open(comfy_log, "w", encoding="utf-8")
        comfy_proc = subprocess.Popen(
            comfy_cmd,
            cwd=str(COMFY_DIR),
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _child_procs.append(comfy_proc)
        log_step(f"Started ComfyUI (PID {comfy_proc.pid}) on 127.0.0.1:{COMFY_PORT}.")

        ready = False
        for _ in range(180):
            if comfy_proc.poll() is not None:
                tail = comfy_log.read_text(errors="ignore")[-1500:] if comfy_log.is_file() else ""
                raise RuntimeError(f"ComfyUI exited early ({comfy_proc.returncode}): {tail[-400:]}")
            try:
                r = requests.get(f"http://127.0.0.1:{COMFY_PORT}/object_info", timeout=3.0)
                if r.status_code == 200:
                    info = r.json()
                    if PRESET == "qwen_h3":
                        if "MiniMaxH3ImageToVideo" in info and "TextEncodeQwenImage21" in info:
                            if pose_installer is not None:
                                pose_installer.verify()
                            ready = True
                            break
                    else:
                        ready = True
                        break
            except Exception:
                pass
            time.sleep(1.0)

        if not ready:
            raise TimeoutError(f"Timed out waiting for ComfyUI nodes on port {COMFY_PORT}")

        public_url = _start_cloudflare_tunnel() if ENABLE_CF_TUNNEL else None

        stage = "models"
        bg_thread: Optional[threading.Thread] = None
        if PRESET == "qwen_h3" and tpl_workflow is not None:
            if BG_MODEL_SYNC:
                log_step("WebUI is up; starting background sync for Qwen-Image-2.1 BF16 + MiniMax H3 weights...")
                bg_thread = threading.Thread(
                    target=_sync_qwen_h3_models,
                    args=(tpl_workflow, False),
                    daemon=True,
                )
                bg_thread.start()
            else:
                _sync_qwen_h3_models(tpl_workflow, update_run_status=True)

        gpu = query_gpu_summary()
        outputs: dict[str, Any] = {
            "preset": PRESET,
            "gpu": gpu.get("name", "unknown"),
            "vram_total_mib": int(gpu.get("vram_total_mb", 0)),
            "workflows_count": len(installed_workflows),
            "primary_workflow": PRIMARY_QWEN_H3_WORKFLOW if PRESET == "qwen_h3" else "default",
            "model_sync": "background" if (PRESET == "qwen_h3" and BG_MODEL_SYNC) else "ready",
        }
        if public_url:
            outputs["cloudflare_url"] = public_url

        setup_done.set()
        ready_msg = (
            f"ComfyUI ready on {gpu.get('name', 'GPU')} · {len(installed_workflows)} workflows loaded"
            + (" (syncing BF16 weights in background)" if (PRESET == "qwen_h3" and BG_MODEL_SYNC) else "")
        )
        write_status("ready", "models", ready_msg, progress=1.0, outputs=outputs)
        log_step("Ready! Monitoring ComfyUI...")

        while not _stop_event.wait(10.0):
            if comfy_proc.poll() is not None:
                fail("server", f"ComfyUI exited unexpectedly (code {comfy_proc.returncode})")
    except SystemExit:
        raise
    except Exception as exc:
        fail(stage, str(exc))


QWEN_H3_CELLS_B64 = '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'

if __name__ == "__main__":
    main()
