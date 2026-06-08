"""
Install NVIDIA LocateAnything into this project (project1 only).

- Clones NVlabs/Eagle -> ./third_party/Eagle
- Installs Embodied package into the active conda env
- Downloads nvidia/LocateAnything-3B weights -> ./.cache/huggingface

Usage:
  conda activate sam3
  export http_proxy="http://127.0.0.1:7893"
  export https_proxy="http://127.0.0.1:7893"
  python scripts/install_locateanything.py
  python scripts/install_locateanything.py --verify-only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EAGLE_DIR = ROOT / "third_party" / "Eagle"
EMBODIED_DIR = EAGLE_DIR / "Embodied"
HF_CACHE = ROOT / ".cache" / "huggingface"
MODEL_ID = "nvidia/LocateAnything-3B"
EAGLE_REPO = "https://github.com/NVlabs/Eagle.git"


def run(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    print(f"[cmd] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd or ROOT, env=env, check=True)


def project_env() -> dict:
    env = os.environ.copy()
    env["HF_HOME"] = str(HF_CACHE)
    env["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE / "hub")
    env.setdefault("PYTHONPATH", "")
    embodied = str(EMBODIED_DIR)
    if embodied not in env["PYTHONPATH"].split(os.pathsep):
        env["PYTHONPATH"] = embodied + (
            os.pathsep + env["PYTHONPATH"] if env["PYTHONPATH"] else ""
        )
    return env


def clone_eagle() -> None:
    if (EMBODIED_DIR / "locateanything_worker.py").exists():
        print(f"[skip] Eagle already present: {EAGLE_DIR}")
        patch_worker_offline_support()
        return
    EAGLE_DIR.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", EAGLE_REPO, str(EAGLE_DIR)])
    patch_worker_offline_support()


def patch_worker_offline_support() -> None:
    """Patch upstream worker to support offline loading from local HF cache."""
    worker_path = EMBODIED_DIR / "locateanything_worker.py"
    if not worker_path.exists():
        return
    text = worker_path.read_text(encoding="utf-8")
    if "local_files_only: bool = False" in text:
        print("[skip] locateanything_worker offline patch already applied")
        return

    old = '''    def __init__(self, model_path: str, device: str = "cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device).eval()'''

    new = '''    def __init__(self, model_path: str, device: str = "cuda", dtype=torch.bfloat16, local_files_only: bool = False):
        self.device = device
        self.dtype = dtype

        load_kw = {"trust_remote_code": True}
        if local_files_only:
            load_kw["local_files_only"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **load_kw)
        self.processor = AutoProcessor.from_pretrained(model_path, **load_kw)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            **load_kw,
        ).to(device).eval()'''

    if old not in text:
        print("[warn] locateanything_worker.py changed upstream; offline patch not applied")
        return
    worker_path.write_text(text.replace(old, new), encoding="utf-8")
    print("[ok] patched locateanything_worker for offline/local cache loading")


def pip_install_embodied() -> None:
    run(
        [sys.executable, "-m", "pip", "install", "-e", str(EMBODIED_DIR)],
        env=project_env(),
    )


def verify_import() -> None:
    env = project_env()
    code = """
import sys
from pathlib import Path
embodied = Path(r"{embodied}")
if str(embodied) not in sys.path:
    sys.path.insert(0, str(embodied))
from locateanything_worker import LocateAnythingWorker
print("import OK:", LocateAnythingWorker)
""".format(embodied=EMBODIED_DIR)
    run([sys.executable, "-c", code], env=env)


def verify_model_load(device: str = "cpu") -> None:
    env = project_env()
    code = f"""
import sys
from pathlib import Path
embodied = Path(r"{EMBODIED_DIR}")
sys.path.insert(0, str(embodied))
from locateanything_worker import LocateAnythingWorker
print("Loading {MODEL_ID} (device={device}) ...")
worker = LocateAnythingWorker("{MODEL_ID}", device="{device}")
print("model load OK")
"""
    run([sys.executable, "-c", code], env=env)


def verify_project_wrapper() -> None:
    sys.path.insert(0, str(ROOT))
    os.environ["HF_HOME"] = str(HF_CACHE)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE / "hub")
    from models.localizer import LocateAnythingWrapper

    wrapper = LocateAnythingWrapper(model_id=MODEL_ID, device="cpu")
    if wrapper.worker is None:
        raise RuntimeError("LocateAnythingWrapper.worker is None after install")
    print("LocateAnythingWrapper OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Install LocateAnything under project1")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--skip-model-download", action="store_true")
    parser.add_argument("--device", default="cpu", help="Device for model load test")
    args = parser.parse_args()

    HF_CACHE.mkdir(parents=True, exist_ok=True)
    print(f"Project root: {ROOT}")
    print(f"Eagle dir:    {EAGLE_DIR}")
    print(f"HF cache:     {HF_CACHE}")

    if args.verify_only:
        verify_import()
        if not args.skip_model_download:
            verify_model_load(device=args.device)
        verify_project_wrapper()
        print("\nAll checks passed.")
        return

    clone_eagle()
    pip_install_embodied()
    verify_import()
    if not args.skip_model_download:
        verify_model_load(device=args.device)
    verify_project_wrapper()
    print("\nLocateAnything installed under project1. Restart web_demo.py to use it.")


if __name__ == "__main__":
    main()
