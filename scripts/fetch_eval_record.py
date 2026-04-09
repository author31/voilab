# rsync -avzP src dest
# call kitchen_registry.is_episode_completed()

# 1. use "rsync -avzP SRC DEST" to get file (only *.ckpt).

# 2. for each ckpt, run scripts/run_isaacsim_eval.py with that ckpt
#    use:
#       a. is_episode_completed()
#       b. timeout=90s
#    to check how many episodes are successful within the timeout.
#    and record the screen while running the evaluation and save the video to DEST


# ===============================================================================================================
# note: you should set up ssh key-based authentication to avoid password prompt before using rsync in the script.
# ===============================================================================================================
# note: this script cannot kill the launching isaac sim after pressing ctrl + c, so you have to kill it manually.
# ===============================================================================================================

#!/usr/bin/env python3
"""
Fetch checkpoints from remote server, run Isaac Sim evaluation with screen recording.

Usage:
    python scripts/fetch_eval_record.py

Features:
    1. Uses rsync to fetch *.ckpt files from remote server
    2. For each checkpoint, runs evaluation with 90s timeout per episode
    3. Records screen during evaluation using mss
    4. Uses kitchen_registry.is_episode_completed() to check success
    5. Saves results summary and videos to DEST directory
"""

import os
import sys
import json
import time
import subprocess
import signal
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import registry


def cleanup_docker_resources():
    """
    Comprehensive cleanup of Docker containers and volumes.
    Uses red text to warn if cleanup is incomplete.
    This function is called on timeout, error, and success to prevent storage explosion.
    """
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    compose_file_dir = Path(__file__).parent.parent

    print("[cleanup] Stopping containers and removing volumes...")

    # Step 1: Stop and remove containers + volumes (all services)
    result = subprocess.run(
        ["docker", "compose", "down", "-v", "--remove-orphans"],
        cwd=compose_file_dir,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"{RED}[cleanup WARNING] docker compose down failed:{RESET}")
        print(f"{RED}  stderr: {result.stderr[:500]}{RESET}")

        # Fallback: force kill and rm
        print("[cleanup] Attempting force cleanup...")
        subprocess.run(
            ["docker", "compose", "kill"], cwd=compose_file_dir, capture_output=True
        )
        subprocess.run(
            ["docker", "compose", "rm", "-f"], cwd=compose_file_dir, capture_output=True
        )

    # Step 2: Prune dangling volumes (as requested)
    print("[cleanup] Pruning dangling volumes...")
    prune_result = subprocess.run(
        ["docker", "volume", "prune", "-f"], capture_output=True, text=True
    )

    if prune_result.returncode != 0:
        print(f"{RED}[cleanup WARNING] Volume prune failed:{RESET}")
        print(f"{RED}  stderr: {prune_result.stderr}{RESET}")

    # Check for remaining volumes that might be related
    vol_result = subprocess.run(
        ["docker", "volume", "ls", "-q"], capture_output=True, text=True
    )

    volumes_count = len([v for v in vol_result.stdout.strip().split("\n") if v])
    if volumes_count > 0:
        print(f"{YELLOW}[cleanup INFO] {volumes_count} volumes still exist{RESET}")
        print(
            f"{YELLOW}  Run 'docker volume ls' and 'docker volume rm <id>' to clean manually if needed{RESET}"
        )

    print("[cleanup] Docker cleanup completed")


def clean(signum, frame):
    print("\n[signal] SIGINT received, cleaning up Docker resources...")
    cleanup_docker_resources()
    sys.exit(1)


# Configuration
SRC = "user@140.113.203.198:/home/user/author_workdir/voilab/data/outputs"
DEST = "/home/hcis-s25/etra/vl-internal-evaluation/ckpteval"
TASK = "kitchen"
TIMEOUT_SECONDS = 180
CONTAINER_DIR = "/workspace/voilab"
EVAL_SCRIPT = CONTAINER_DIR + "/scripts/run_isaacsim_eval.py"

# Ensure DEST exists
os.makedirs(DEST, exist_ok=True)


def needs_rsync() -> bool:
    """Check if rsync would transfer any files (using dry-run with modification time check)."""
    cmd = [
        "rsync",
        "-avz",
        "--dry-run",
        "--out-format=%f",
        SRC + "/",
        DEST + "/",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        # Parse output - lines that don't end with '/' are files to be transferred
        # These include new files, modified files (different size/time), or deleted files
        transfer_lines = [
            line
            for line in result.stdout.split("\n")
            if line.strip()
            and not line.endswith("/")
            and not line.startswith("skipping")
        ]
        return len(transfer_lines) > 0
    except Exception as e:
        print(f"[rsync check] Error checking sync status: {e}")
        return True  # Assume sync needed on error


def run_rsync():
    """Run rsync to fetch checkpoint files from remote server."""
    print(f"[rsync] Fetching checkpoints from {SRC}...")

    # Build rsync command to only fetch .ckpt files
    cmd = [
        "rsync",
        "-avzP",
        SRC + "/",
        DEST + "/",
    ]

    try:
        result = subprocess.run(
            cmd,
            # capture_output=True,
            text=True,
            check=True,
        )
        print(f"[rsync] Success: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[rsync] Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print("[rsync] Error: rsync command not found. Please install rsync.")
        return False


def run_rsync_if_needed():
    """Only run rsync if there are files that need updating (new or modified)."""
    print(
        "[rsync] Checking if transfer is needed (comparing files and modification times)..."
    )

    if not needs_rsync():
        print(
            f"\033[95m[rsync] All files are up to date (same modification times), skipping transfer.\033[0m"
        )
        return True

    print("[rsync] New or modified files detected, starting transfer...")
    return run_rsync()


def get_checkpoint_files() -> List[Path]:
    """Get all checkpoint files in DEST directory."""
    dest_path = Path(DEST)
    ckpt_files = list(dest_path.rglob("*.ckpt"))
    return sorted(ckpt_files)


def setup_screen_recorder(output_path: str, fps: int = 10):
    """Setup screen recorder using mss library."""
    try:
        import mss
        import numpy as np
        import cv2
    except ImportError as e:
        print(f"[screen recorder] Error: Missing required library: {e}")
        print(
            "[screen recorder] Please install: pip install mss opencv-python-headless"
        )
        return None

    class ScreenRecorder:
        def __init__(self, output_path: str, fps: int = 10):
            self.output_path = output_path
            self.fps = fps
            self.frame_interval = 1.0 / fps
            self.recording = False
            self.frames: List[Any] = []
            self.thread: Optional[threading.Thread] = None
            self._stop_event = threading.Event()

        def start(self):
            self.recording = True
            self._stop_event.clear()
            self.thread = threading.Thread(target=self._record)
            self.thread.start()

        def _record(self):
            import mss
            import numpy as np

            # Create mss instance INSIDE the recording thread
            sct = mss.mss()
            monitor = sct.monitors[1]  # Primary monitor

            time.sleep(90) # wait for isaac sim launch

            print(f"[screen recorder] Started recording to {self.output_path}")

            try:
                while self.recording and not self._stop_event.is_set():
                    start_time = time.time()

                    # Capture screen
                    screenshot = sct.grab(monitor)
                    # Convert to numpy array
                    frame = np.array(screenshot)
                    # mss returns BGRA, convert to RGB
                    frame = frame[:, :, :3]
                    self.frames.append(frame)

                    # Maintain frame rate
                    elapsed = time.time() - start_time
                    sleep_time = self.frame_interval - elapsed
                    if sleep_time > 0:
                        # Use event.wait() instead of time.sleep() for responsive stopping
                        self._stop_event.wait(timeout=sleep_time)
            finally:
                # Always close mss in the same thread where it was created
                try:
                    sct.close()
                except Exception:
                    # Ignore any errors during close (display may already be invalid)
                    pass

        def stop(self):
            if not self.recording:
                return

            self.recording = False
            self._stop_event.set()

            if self.thread:
                self.thread.join(timeout=3.0)
                if self.thread.is_alive():
                    print(
                        "[screen recorder] Warning: Recording thread did not stop gracefully"
                    )

            # Save video (mss already closed in _record thread)
            self._save_video()

        def _save_video(self):
            import cv2

            if not self.frames:
                print("[screen recorder] No frames captured")
                return

            # Get frame dimensions from first frame
            height, width = self.frames[0].shape[:2]

            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))

            for frame in self.frames:
                # Convert RGB to BGR for OpenCV
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)

            out.release()
            print(
                f"[screen recorder] Saved video: {self.output_path} ({len(self.frames)} frames)"
            )

    return ScreenRecorder(output_path, fps)


class TimeoutEvaluator:
    """Evaluator with 90s timeout per episode using is_episode_completed()."""

    def __init__(self, task_name: str, timeout_seconds: int = 90):
        self.task_name = task_name
        self.timeout_seconds = timeout_seconds
        self.registry_class = registry.get_task_registry(task_name)
        self.is_episode_completed = self.registry_class.is_episode_completed

    def evaluate_checkpoint(
        self, ckpt_path: Path, output_dir: Path, device: str = "cuda:0"
    ) -> Dict[str, Any]:
        """Run evaluation with timeout and screen recording."""

        ckpt_name = ckpt_path.stem
        # HARD CODING FOR DEBUGGING - adjust container checkpoint path based on your setup
        host_base = DEST
        # container_ckpt_path = f"{CONTAINER_DIR}/ckpteval/23.46.49_train_diffusion_unet_timm_umi/checkpoints/{ckpt_name}.ckpt"
        container_ckpt_path = str(ckpt_path).replace(
            host_base, f"{CONTAINER_DIR}/ckpteval"
        )

        # Define paths to check
        video_path = output_dir / f"{ckpt_name}_screen_record.mp4"
        eval_output_dir = output_dir / ckpt_name

        # Check if BOTH video AND eval log already exist
        if video_path.exists():
            print(f"\033[95m[eval] Skipping evaluation for {ckpt_name} because file already exist\033[0m")

            # Load existing success rate from eval_log.json
            success_rate = self._get_success_rate_from_log(eval_output_dir)

            return {
                "checkpoint": str(ckpt_path),
                "checkpoint_name": ckpt_name,
                "skipped": True,
                "video_path": str(video_path),
                "eval_output_dir": str(eval_output_dir),
                "success": True,
                "success_rate": success_rate,
            }

        # =======================
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {ckpt_name}")
        print(f"Checkpoint: {ckpt_path}\nContainer path: {container_ckpt_path}")
        print(f"Output dir: {output_dir}")
        print(f"Container dir: {CONTAINER_DIR}")
        print(f"Timeout: {self.timeout_seconds}s per episode")
        print(f"{'=' * 60}")

        # Setup screen recorder
        screen_recorder = setup_screen_recorder(str(video_path), fps=10)

        # Build evaluation command to run inside Docker container
        cmd = [
            "docker",
            "compose",
            "run",
            "-it",
            "--rm",
            "isaac-sim",
            "/bin/bash",
            "-c",
            f"/workspace/voilab/.venv/bin/python {EVAL_SCRIPT} \
                --task {self.task_name} --checkpoint {container_ckpt_path} \
                --output-dir {eval_output_dir} --device {device}",
        ]

        print(f"[eval] Running command: {' '.join(cmd)}")

        # Start screen recording
        if screen_recorder:
            screen_recorder.start()

        # Run evaluation with timeout
        start_time = time.time()
        try:
            # Set up environment
            env = os.environ.copy()

            # Run with timeout
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )

            try:
                stdout, stderr = process.communicate(timeout=self.timeout_seconds)
                print("STDOUT:", stdout)
                print("STDERR:", stderr)
                return_code = process.returncode
            except subprocess.TimeoutExpired:
                print(
                    f"[eval] Timeout after {self.timeout_seconds}s - killing process group and cleaning up Docker resources"
                )
                # Kill the entire process group
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)

                # Always clean up Docker resources on timeout
                cleanup_docker_resources()

                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                stdout, stderr = "", ""
                return_code = -1

        except Exception as e:
            print(f"[eval] Error running evaluation: {e}")
            stdout, stderr = "", str(e)
            return_code = -1
        finally:
            # Stop screen recording
            if screen_recorder:
                screen_recorder.stop()

            # Cleanup Docker resources to prevent storage explosion
            # This runs on success, timeout, and error cases
            cleanup_docker_resources()

        elapsed = time.time() - start_time
        print(
            f"[eval] Evaluation completed in {elapsed:.1f}s (return code: {return_code})"
        )

        # Parse results
        results = self._parse_results(eval_output_dir, stdout, stderr, return_code)
        results["checkpoint"] = str(ckpt_path)
        results["checkpoint_name"] = ckpt_name
        results["elapsed_seconds"] = elapsed
        results["video_path"] = video_path if screen_recorder else None

        return results

    @staticmethod
    def _get_success_rate_from_log(eval_output_dir: Path) -> Optional[float]:
        """Extract success rate from eval_log.json after evaluation."""
        eval_log_path = eval_output_dir / "eval_log.json"
        if not eval_log_path.exists():
            return None

        try:
            with open(eval_log_path, "r") as f:
                eval_log = json.load(f)
            return eval_log.get("success_rate")
        except Exception:
            return None

    def _parse_results(
        self, eval_output_dir: Path, stdout: str, stderr: str, return_code: int
    ) -> Dict[str, Any]:
        """Parse evaluation results from output directory and logs."""
        results = {
            "success": return_code == 0,
            "return_code": return_code,
            "stdout": stdout[-5000:] if stdout else "",  # Last 5000 chars
            "stderr": stderr[-5000:] if stderr else "",
        }

        # Try to load eval_log.json if it exists
        eval_log_path = eval_output_dir / "eval_log.json"
        if eval_log_path.exists():
            try:
                with open(eval_log_path, "r") as f:
                    eval_log = json.load(f)
                results["eval_log"] = eval_log
                results["success_rate"] = eval_log.get("success_rate", None)
            except Exception as e:
                print(f"[eval] Warning: Could not parse eval_log.json: {e}")
                results["eval_log_error"] = str(e)

        # Also try to find and parse any success information from stdout
        if "success_rate" not in results or results["success_rate"] is None:
            # Parse success rate from stdout
            for line in stdout.split("\n"):
                if "Success rate:" in line:
                    try:
                        success_rate = float(
                            line.split(":")[-1].strip().replace("%", "")
                        )
                        results["success_rate"] = success_rate
                    except ValueError:
                        pass
                    break

        return results


def save_summary(results_list: List[Dict[str, Any]], output_dir: Path):
    """Save evaluation summary to JSON file."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "task": TASK,
        "timeout_seconds": TIMEOUT_SECONDS,
        "total_checkpoints": len(results_list),
        "results": results_list,
    }

    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"Evaluation Summary:")
    print(f"{'=' * 60}")
    print(f"Total checkpoints evaluated: {len(results_list)}")

    for r in results_list:
        ckpt_name = r.get("checkpoint_name", "unknown")
        if r.get("skipped", False):
            success_rate = r.get("success_rate", "N/A")
            print(f"  {ckpt_name}: skipped (previous success_rate={success_rate})")
        else:
            success = r.get("success", False)
            success_rate = r.get("success_rate", "N/A")
            elapsed = r.get("elapsed_seconds", 0)
            print(
                f"  {ckpt_name}: success={success}, success_rate={success_rate}, time={elapsed:.1f}s"
            )

    print(f"\nFull summary saved to: {summary_path}")


def main():
    signal.signal(signal.SIGINT, clean)  # Handle Ctrl+C gracefully
    print("=" * 60)
    print("Checkpoint Fetch and Evaluation Script")
    print("=" * 60)
    print(f"Source: {SRC}")
    print(f"Destination: {DEST}")
    print(f"Task: {TASK}")
    print(f"Episode timeout: {TIMEOUT_SECONDS}s")
    print("=" * 60)

    # Step 1: Fetch checkpoints from remote (only if needed)
    print("\n[Step 1] Checking/Fetching checkpoints from remote server...")
    if not run_rsync_if_needed():
        print("[Error] Failed to fetch checkpoints. Exiting.")
        return 1

    # Step 2: Get list of checkpoint files
    print("\n[Step 2] Finding checkpoint files...")
    ckpt_files = get_checkpoint_files()

    if not ckpt_files:
        print(f"[Warning] No checkpoint files found in {DEST}")
        return 0

    print(f"Found {len(ckpt_files)} checkpoint(s):")
    for ckpt in ckpt_files:
        print(f"  - {ckpt.name}")

    # Step 3: Evaluate each checkpoint
    print(f"\n[Step 3] Running evaluations...")
    evaluator = TimeoutEvaluator(TASK, timeout_seconds=TIMEOUT_SECONDS)
    results_list: List[Dict[str, Any]] = []

    output_dir = Path(DEST)

    for i, ckpt_path in enumerate(ckpt_files, 1):
        print(f"\n[{i}/{len(ckpt_files)}] Processing {ckpt_path.name}...")

        try:
            result = evaluator.evaluate_checkpoint(ckpt_path, output_dir)
            results_list.append(result)
        except Exception as e:
            print(f"[Error] Failed to evaluate {ckpt_path.name}: {e}")
            import traceback

            traceback.print_exc()
            results_list.append(
                {
                    "checkpoint": str(ckpt_path),
                    "checkpoint_name": ckpt_path.stem,
                    "error": str(e),
                    "success": False,
                }
            )

    # Step 4: Save summary
    print("\n[Step 4] Saving summary...")
    save_summary(results_list, output_dir)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
