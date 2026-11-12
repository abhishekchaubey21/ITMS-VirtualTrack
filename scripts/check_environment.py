"""Environment verification script for ITMS-VirtualTrack (Milestone 1).

Inspects the Windows/Python/Blender environment, locates the Blender executable,
verifies Blender headless API support, checks Python dependencies, GPU/CUDA,
FFmpeg, disk space, and validates project structure and configuration files.
"""

from __future__ import annotations

import argparse
import base64
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_DIRS = [
    "config",
    "simulation/blender",
    "simulation/python",
    "dataset/synthetic/videos",
    "dataset/synthetic/images/train",
    "dataset/synthetic/images/val",
    "dataset/synthetic/images/test",
    "dataset/synthetic/labels/train",
    "dataset/synthetic/labels/val",
    "dataset/synthetic/labels/test",
    "dataset/synthetic/ground_truth",
    "dataset/synthetic/metadata",
    "detection",
    "evaluation",
    "dashboard",
    "results/videos",
    "results/graphs",
    "results/metrics",
    "results/reports",
    "scripts",
    "docs",
]

REQUIRED_CONFIGS = [
    "config/simulation_config.json",
    "config/camera_config.json",
    "config/anomaly_config.json",
    "config/rendering_config.json",
]

PYTHON_PACKAGES = [
    ("numpy", "core", True),
    ("pandas", "core", True),
    ("scipy", "core", True),
    ("yaml", "core", True),
    ("cv2", "vision", True),
    ("PIL", "vision", True),
    ("torch", "dl", True),
    ("torchvision", "dl", True),
    ("ultralytics", "dl", True),
    ("sklearn", "evaluation", True),
    ("tqdm", "utility", True),
    ("matplotlib", "visualization", True),
    ("plotly", "visualization", False),
    ("streamlit", "dashboard", False),
    ("pytest", "testing", False),
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def check_directories() -> dict[str, Any]:
    missing = [rel for rel in REQUIRED_DIRS if not (PROJECT_ROOT / rel).is_dir()]
    return {"ok": len(missing) == 0, "missing": missing}


def check_configs() -> dict[str, Any]:
    errors: dict[str, str] = {}
    parsed: dict[str, dict[str, Any]] = {}
    for rel in REQUIRED_CONFIGS:
        cfg_path = PROJECT_ROOT / rel
        if not cfg_path.is_file():
            errors[rel] = "File not found"
            continue
        try:
            parsed[rel] = load_json(cfg_path)
        except Exception as exc:
            errors[rel] = f"Invalid JSON: {exc}"
    return {"ok": len(errors) == 0, "errors": errors, "configs": parsed}


def find_blender_executable(
    cli_path: str | None, sim_cfg: dict[str, Any] | None
) -> str | None:
    candidates: list[str] = []
    if cli_path:
        candidates.append(cli_path)

    env_path = os.environ.get("BLENDER_PATH")
    if env_path:
        candidates.append(env_path)

    if sim_cfg:
        paths_cfg = sim_cfg.get("paths", {})
        configured = paths_cfg.get("blender_executable")
        if configured:
            candidates.append(configured)
        candidates.extend(paths_cfg.get("blender_search_candidates", []))

    which_blender = shutil.which("blender")
    if which_blender:
        candidates.append(which_blender)

    # Common fallback paths on Windows
    candidates.extend(
        [
            r"D:\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 5.3\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
        ]
    )

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        p = Path(candidate)
        if p.is_file():
            return str(p)
    return None


def check_blender(blender_exe: str | None, verify_api: bool = True) -> dict[str, Any]:
    if not blender_exe:
        return {
            "found": False,
            "path": None,
            "version": None,
            "api_ok": False,
            "error": (
                "Blender executable not found. Set BLENDER_PATH or "
                "paths.blender_executable in config/simulation_config.json."
            ),
        }

    try:
        ver_proc = subprocess.run(
            [blender_exe, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:
        return {
            "found": True,
            "path": blender_exe,
            "version": None,
            "api_ok": False,
            "error": f"Failed to execute Blender: {exc}",
        }

    if ver_proc.returncode != 0:
        return {
            "found": True,
            "path": blender_exe,
            "version": None,
            "api_ok": False,
            "error": (ver_proc.stderr or ver_proc.stdout).strip(),
        }

    first_line = (ver_proc.stdout or "").splitlines()[0].strip()
    result: dict[str, Any] = {
        "found": True,
        "path": blender_exe,
        "version": first_line,
        "api_ok": False,
        "blender_python": None,
        "engines_tested": {},
        "video_support": None,
        "error": None,
    }

    if not verify_api:
        result["api_ok"] = True
        return result

    probe_script = "\n".join(
        [
            "import bpy, json, sys",
            "s = bpy.context.scene",
            "s.render.engine = 'BLENDER_EEVEE'",
            "eevee_ok = (s.render.engine == 'BLENDER_EEVEE')",
            "try:",
            "    s.render.engine = 'CYCLES'",
            "    cycles_ok = (s.render.engine == 'CYCLES')",
            "except Exception:",
            "    cycles_ok = False",
            "s.render.image_settings.media_type = 'VIDEO'",
            "payload = {",
            "    'py': sys.version.split()[0],",
            "    'eevee': eevee_ok,",
            "    'cycles': cycles_ok,",
            "    'file_format': s.render.image_settings.file_format,",
            "}",
            "print('ITMS_PROBE:' + json.dumps(payload))",
        ]
    )
    encoded_probe = base64.b64encode(probe_script.encode("utf-8")).decode("ascii")
    probe_expr = f"import base64; exec(base64.b64decode('{encoded_probe}').decode('utf-8'))"
    try:
        api_proc = subprocess.run(
            [blender_exe, "-b", "--factory-startup", "--python-expr", probe_expr],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        probe_line = None
        for line in (api_proc.stdout or "").splitlines():
            if line.startswith("ITMS_PROBE:"):
                probe_line = line[len("ITMS_PROBE:") :]
                break
        if api_proc.returncode == 0 and probe_line:
            probe_data = json.loads(probe_line)
            result["blender_python"] = probe_data.get("py")
            result["engines_tested"] = {
                "BLENDER_EEVEE": probe_data.get("eevee"),
                "CYCLES": probe_data.get("cycles"),
            }
            result["video_support"] = probe_data.get("file_format")
            result["api_ok"] = bool(probe_data.get("eevee"))
        else:
            result["error"] = (
                f"Blender headless probe failed (rc={api_proc.returncode}): "
                f"{(api_proc.stderr or api_proc.stdout).strip()}"
            )
    except Exception as exc:
        result["error"] = f"Blender headless probe error: {exc}"

    return result


def check_python_environment() -> dict[str, Any]:
    ver_info = sys.version_info
    py_ok = (ver_info.major, ver_info.minor) >= (3, 10)
    installed: dict[str, str] = {}
    missing_required: list[str] = []
    missing_optional: list[str] = []

    for mod_name, _category, required in PYTHON_PACKAGES:
        try:
            mod = importlib.import_module(mod_name)
            installed[mod_name] = str(getattr(mod, "__version__", "installed"))
        except Exception:
            if required:
                missing_required.append(mod_name)
            else:
                missing_optional.append(mod_name)

    cuda_info: dict[str, Any] = {"available": False, "device": None}
    try:
        import torch

        cuda_avail = bool(torch.cuda.is_available())
        cuda_info["available"] = cuda_avail
        if cuda_avail:
            cuda_info["device"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return {
        "executable": sys.executable,
        "version": platform.python_version(),
        "version_ok": py_ok,
        "installed": installed,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "cuda": cuda_info,
    }


def check_ffmpeg() -> dict[str, Any]:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return {"found": False, "path": None, "version": None}
    try:
        proc = subprocess.run(
            [ffmpeg_path, "-version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        first_line = (proc.stdout or "").splitlines()[0].strip() if proc.stdout else None
        return {"found": True, "path": ffmpeg_path, "version": first_line}
    except Exception:
        return {"found": True, "path": ffmpeg_path, "version": None}


def check_disk_space() -> dict[str, Any]:
    drives: dict[str, dict[str, float]] = {}
    for target in [Path("C:/"), PROJECT_ROOT]:
        anchor = target.anchor or str(target)
        if anchor in drives:
            continue
        if Path(anchor).exists():
            usage = shutil.disk_usage(anchor)
            drives[anchor] = {
                "total_gb": round(usage.total / (1024**3), 2),
                "free_gb": round(usage.free / (1024**3), 2),
            }
    return drives


def run_checks(blender_arg: str | None, verify_blender_api: bool) -> dict[str, Any]:
    dirs_res = check_directories()
    cfg_res = check_configs()
    sim_cfg = cfg_res["configs"].get("config/simulation_config.json")
    blender_exe = find_blender_executable(blender_arg, sim_cfg)
    blender_res = check_blender(blender_exe, verify_api=verify_blender_api)
    py_res = check_python_environment()
    ffmpeg_res = check_ffmpeg()
    disk_res = check_disk_space()

    core_ready = (
        dirs_res["ok"]
        and cfg_res["ok"]
        and blender_res["found"]
        and blender_res["api_ok"]
        and py_res["version_ok"]
    )
    return {
        "project_root": str(PROJECT_ROOT),
        "platform": platform.platform(),
        "core_ready_for_simulation": core_ready,
        "directories": dirs_res,
        "configs": {
            "ok": cfg_res["ok"],
            "errors": cfg_res["errors"],
        },
        "blender": blender_res,
        "python": py_res,
        "ffmpeg": ffmpeg_res,
        "disk_gb": disk_res,
    }


def print_human_report(report: dict[str, Any]) -> None:
    print("=" * 68)
    print("ITMS-VirtualTrack — Environment Check (SYNTHETIC SIMULATION PLATFORM)")
    print("=" * 68)
    print(f"Project Root : {report['project_root']}")
    print(f"Platform     : {report['platform']}")
    print()

    dirs = report["directories"]
    print(f"[1] Directory Structure : {'OK' if dirs['ok'] else 'MISSING'}")
    if dirs["missing"]:
        for m in dirs["missing"]:
            print(f"    - Missing: {m}")

    cfgs = report["configs"]
    print(f"[2] Config JSON Files   : {'OK' if cfgs['ok'] else 'ERRORS'}")
    for name, err in cfgs["errors"].items():
        print(f"    - {name}: {err}")

    bl = report["blender"]
    print(f"[3] Blender Executable  : {bl['path'] or 'NOT FOUND'}")
    if bl["found"]:
        print(f"    Version             : {bl['version']}")
        print(f"    Blender Python      : {bl.get('blender_python')}")
        print(f"    Headless API Check  : {'OK' if bl['api_ok'] else 'FAILED'}")
        if bl.get("engines_tested"):
            print(f"    Render Engines      : {bl['engines_tested']}")
        if bl.get("video_support"):
            print(f"    Video Format        : {bl['video_support']}")
    if bl.get("error"):
        print(f"    Error               : {bl['error']}")

    py = report["python"]
    print(f"[4] Host Python         : {py['version']} ({py['executable']})")
    print(f"    CUDA Available      : {py['cuda']['available']} ({py['cuda']['device']})")
    print(f"    Installed Packages  : {len(py['installed'])} checked")
    if py["missing_required"]:
        print(f"    MISSING (Required)  : {', '.join(py['missing_required'])}")
    if py["missing_optional"]:
        print(f"    MISSING (Optional)  : {', '.join(py['missing_optional'])}")

    ff = report["ffmpeg"]
    print(f"[5] FFmpeg              : {ff['path'] or 'NOT FOUND'}")
    if ff["version"]:
        print(f"    Version             : {ff['version']}")

    print("[6] Disk Space:")
    for drive, stats in report["disk_gb"].items():
        warn = " [LOW SPACE WARNING]" if stats["free_gb"] < 8.0 else ""
        print(f"    {drive:<6} Free: {stats['free_gb']} GB / Total: {stats['total_gb']} GB{warn}")

    print()
    print(
        "Core Simulation Readiness: "
        + ("READY (Milestone 1 verified)" if report["core_ready_for_simulation"] else "NOT READY")
    )
    print("=" * 68)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify ITMS-VirtualTrack environment and dependencies."
    )
    parser.add_argument(
        "--blender",
        default=None,
        help="Optional explicit path to blender.exe.",
    )
    parser.add_argument(
        "--skip-blender-api",
        action="store_true",
        help="Skip invoking Blender headless API probe.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON report instead of formatted text.",
    )
    args = parser.parse_args()

    report = run_checks(
        blender_arg=args.blender,
        verify_blender_api=not args.skip_blender_api,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_human_report(report)

    return 0 if report["core_ready_for_simulation"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

