# The uv workspace

How this repo declares, locks and installs its dependencies with `uv` — one virtualenv, one lockfile, three packages.

**Read this if:** you have only ever used `pip` or `conda` and need to install, run, or add a dependency here.

**Before you start:** [Getting started](./getting-started.md) if you just want the machine set up; this page explains *why* those commands work.

---

## 1. Why uv instead of a conda environment

Upstream UMI (`real-stanford/universal_manipulation_interface`) ships a conda environment file and asks you to run `mamba env create`, then `pip install -e .` by hand. This fork replaced all of that with `uv`, a single Rust binary that resolves, locks, installs, and runs. There is no conda or mamba environment file anywhere in this repo.

- **One lockfile, hash-pinned.** `uv.lock` (6,549 lines) pins **369** packages with sha256 hashes. Conda YAML files list version ranges, not hashes, so two people running the same file on different days get different environments. Here they do not.
- **Several packages, all editable, one environment.** `voilab`, `umi` and `diffusion_policy` are developed together and installed as links to the source tree, so editing `packages/umi/src/umi/*.py` takes effect on the next `import` — no reinstall, no `PYTHONPATH` juggling.
- **Fast resolution.** Re-checking the whole 369-package graph takes under a second (`uv lock --check`), so re-locking is not something you avoid doing.
- **Awkward dependencies are first-class.** UMI needs packages from git, from tarball URLs, and from a vendored wheel file. `uv` locks all of those the same way it locks PyPI packages, instead of leaving them as README instructions.

There is no `conda activate` step. `uv run <cmd>` syncs the environment and runs the command inside it.

## 2. What a workspace is, in plain words

A **uv workspace** is one directory tree that contains several Python distributions sharing:

- **one virtualenv**, at `.venv/` in the repo root;
- **one lockfile**, `uv.lock`, covering all of them together;
- **one `uv sync`** that installs every member, in editable mode.

"Editable" means the venv holds a pointer to your source directory rather than a copy. You can see the pointers:

```bash
cat /home/hcis-s17/author_workdir/voilab/.venv/lib/python3.10/site-packages/umi.pth
# -> /home/hcis-s17/author_workdir/voilab/packages/umi/src
```

Contrast with conda, where each package you want to hack on needs its own `pip install -e .` and nothing guarantees the resulting set of versions is consistent.

## 3. The three packages

| Distribution name | Path | Import as | Console script | Role |
|---|---|---|---|---|
| `voilab` (0.1.3) | `.` (source in `src/voilab/`) | `voilab` | `voilab` | Root package: Jupyter/Voila viewers and the Isaac Sim launcher CLI |
| `umi` (0.1.0) | `packages/umi` | `umi` | `umi` | The UMI fork: SLAM pipeline stages, GoPro handling, real-robot interfaces |
| `diffusion-policy` (0.1.0) | `packages/diffusion_policy` | `diffusion_policy` | *(none)* | Policy training and evaluation code |

Note the distribution name `diffusion-policy` is hyphenated while the import name is underscored. The `uv_build` backend (`pyproject.toml:1-3`) assumes a `src/` layout and normalises the hyphen automatically, which is why no extra configuration is needed.

Dependency direction — the root depends on both members, never the reverse:

```text
voilab  (root: src/voilab)
  |
  +--> umi                (packages/umi)          --> torch, opencv, ray, py-gpmf-parser, ...
  |
  +--> diffusion_policy   (packages/diffusion_policy) --> huggingface-hub, zarr, ikpy, ...
  |
  +--[dev extra]--> jupyterlab-urdf   (local wheel in deps/)
```

**Known issue:** `packages/diffusion_policy/pyproject.toml:18-24` declares only five dependencies, but `packages/diffusion_policy/train.py` imports `hydra` and `omegaconf` and pulls in torch. It works only because `umi` drags those into the shared venv. Installing `diffusion-policy` on its own would fail at import. See [Known issues](./known-issues.md).

## 4. Where dependencies are declared

The workspace itself is declared once, in the root `pyproject.toml:52-59`:

```toml
[tool.uv.workspace]
members = ["packages/*"]      # every subdir of packages/ holding a pyproject.toml

[tool.uv.sources]
umi = { workspace = true }               # resolve from this repo, not PyPI; install editable
diffusion_policy = { workspace = true }  # same
jupyterlab-urdf = { path = "deps/jupyterlab_urdf-0.6.0-py3-none-any.whl" }
```

`umi` and `diffusion_policy` then appear as ordinary requirement strings in `pyproject.toml:29-30`. `[tool.uv.sources]` is what redirects those names away from PyPI.

Four kinds of non-PyPI dependency are in use:

| Kind | Real example | Declared at | What it needs to build |
|---|---|---|---|
| Workspace member | `umi`, `diffusion_policy` | `pyproject.toml:57-58` | Nothing — linked from the source tree |
| Local path wheel | `jupyterlab-urdf` → `deps/jupyterlab_urdf-0.6.0-py3-none-any.whl` | `pyproject.toml:59` | Nothing; the wheel is vendored and committed to git |
| Git dependency | `py-gpmf-parser @ git+https://github.com/urbste/py-gpmf-parser.git` | `packages/umi/pyproject.toml:85` | **cmake + a C compiler.** No wheels are published, so it compiles a C extension during `uv sync` |
| URL tarball | `spnav @ https://github.com/cheng-chi/spnav/archive/<sha>.tar.gz` and `robosuite @ .../<sha>.tar.gz` | `packages/umi/pyproject.toml:83-84` | A C compiler for `spnav`; both are source archives, not wheels |

Details worth knowing:

- The git and URL entries are **PEP 508 direct references written inline** in the dependency list. `packages/umi/pyproject.toml` has no `[tool.uv]` table at all. Only the local wheel needs `[tool.uv.sources]`, because the bare name `jupyterlab-urdf` would otherwise resolve against PyPI.
- Every one of them still lands in the lockfile with a hash and a pinned commit — for example `uv.lock:3837` pins `py-gpmf-parser` to commit `8af151545c188f55d3d46d78eefe4e5881a9057c`.
- `py-gpmf-parser` reads GoPro telemetry (the IMU stream embedded in the video file). See [GoPro telemetry](./gopro-telemetry.md). It is the concrete reason `make install-cmake` exists.
- `spnav` (3Dconnexion SpaceMouse driver bindings) and `robosuite` (simulation environments) are forks by the original UMI author, carried over unchanged. See [vs upstream UMI](./vs-upstream-umi.md).

## 5. Daily commands

Run all of these from the repo root — the directory holding `pyproject.toml` and `uv.lock`.

| Task | Command |
|---|---|
| Install runtime dependencies | `uv sync` |
| Install runtime + dev tools (pytest, ruff, notebook) | `uv sync --extra dev` |
| Install exactly what is locked, never re-lock | `uv sync --frozen` |
| Run anything inside the venv | `uv run <command>`, e.g. `uv run umi run-slam-pipeline <config.yaml>` |
| Add a dependency to the root package | `uv add <package>` |
| Add a dependency to the `dev` extra | `uv add --optional dev <package>` |
| Add a dependency to a workspace member | `uv add --package umi <package>` (use `--package diffusion-policy` for the hyphenated name) |
| Re-resolve the lockfile without installing | `uv lock` |
| Check the lockfile still matches the pyprojects | `uv lock --check` |

Two things to internalise:

- **`uv.lock` is committed to git.** If your change to any `pyproject.toml` alters the lock, commit the lock in the same commit. `uv lock --check` is what CI-style verification would run; it currently passes.
- **`dev` is an extra, not a dependency group.** It lives in `[project.optional-dependencies]` (`pyproject.toml:34-45`), so the flag is `--extra dev`, not `--group dev`.

`uv sync` does everything in one shot: download CPython if missing, create `.venv/`, resolve against the lock, install all 369 packages, link the three workspace packages editable, and put the `voilab` and `umi` console scripts in `.venv/bin/`. For what those scripts do, see [CLI reference](./cli-reference.md).

## 6. Python version

| Where | Value |
|---|---|
| `.python-version` | `3.10` — what `uv sync` picks by default |
| `pyproject.toml:13` | `requires-python = ">=3.10, <3.13"` |
| `packages/umi/pyproject.toml:16`, `packages/diffusion_policy/pyproject.toml:16` | `>=3.10` — no upper bound, inconsistent with the root |
| `uv.lock:3` | `>=3.10, <3.13` |

**Known issue:** the declared `<3.13` is too generous. `torch==2.1.0` and `torchvision==0.16.0` publish no source distribution and only `cp310`/`cp311` wheels; `ur-rtde==1.5.6` is also `cp310`/`cp311` only, and `numba==0.57` predates 3.12. Locking succeeds because uv resolves per-marker, but `uv sync` on a 3.12 interpreter fails to find a torch distribution with a confusing error. **Use Python 3.10** — the value already in `.python-version` — or 3.11, which is what the Docker images build. See [Known issues](./known-issues.md) and [Simulation and Docker](./simulation-and-docker.md).

## 7. Makefile targets

| Target | What it does | When you need it |
|---|---|---|
| `make install-uv` | Installs `uv` via the official script if `command -v uv` fails. Idempotent (`Makefile:2-10`) | First step on a fresh machine |
| `make install` | `uv sync` (`Makefile:13-16`) | Standard install |
| `make install-dev` | `uv sync --extra dev` (`Makefile:19-22`) | Before running pytest, ruff, or JupyterLab |
| `make launch-jupyterlab` | `uv run jupyter lab --ip 0.0.0.0 --port 8888 --no-browser` (`Makefile:25-28`) | The notebook/Voila workflow — see [Visualization](./visualization.md) |
| `make install-exiftool` | Installs `libimage-exiftool-perl` unless `exiftool -ver` is already ≥ 12.5 (`Makefile:30-44`) | Before running the SLAM pipeline; it reads GoPro metadata |
| `make install-cmake` | `sudo apt install -y cmake`, unconditionally (`Makefile:46-49`) | Before the first `uv sync`, so `py-gpmf-parser` can compile |
| `make install-ffmpeg` | Installs ffmpeg if absent (`Makefile:51-61`) | Video transcoding inside the pipeline |
| `make launch-workspace` | Runs `./launch_workspace.sh` (`Makefile:63-66`) | Starting the dev Docker container |
| `make launch-workspace-force` | Same, with `--force-rebuild` (`Makefile:68-71`) | Rebuilding that image |
| `make init-submodule` | `git submodule update --init --recursive` (`Makefile:73-77`) | Before building the Isaac Sim ROS image |

Three footguns in this file:

- **Known issue:** seven of the ten `.PHONY` lines are written `.PHONY install-uv:` (a space) instead of `.PHONY: install-uv` (a colon) — `Makefile:1,12,18,24,63,68,73`. GNU make reads those as a recipe-less rule with two targets, so those names never actually reach `.PHONY`. `make -p` confirms the real list is only `install-exiftool install-cmake install-ffmpeg`. If a file or directory named `install`, `install-dev` or `launch-workspace` ever appears in the repo root, that target silently stops running. See [Known issues](./known-issues.md).
- **Known issue:** `Makefile:73` declares `make-init-submodule` while the recipe below it belongs to `init-submodule`. Running `make make-init-submodule` succeeds and does nothing. The working target is `make init-submodule`.
- Bare `make` with no target runs `install-uv`, because make skips names starting with `.` when choosing a default goal.

There is no `test`, `lint`, `format` or `clean` target.

## 8. Tests and linting

`pytest` and `ruff` are in the `dev` extra only, so install that first.

```bash
cd /home/hcis-s17/author_workdir/voilab
uv sync --extra dev
uv run pytest packages/umi/tests --ignore=packages/umi/tests/services/test_aruco_detection.py
uv run ruff check src packages
```

There is no root pytest configuration, no `pytest.ini`, and no `conftest.py`. The only config is a `[tool.pytest.ini_options]` block inside each member package (`packages/umi/pyproject.toml:5-9`), setting `minversion = "6.0"` and `pythonpath = ["src"]`. Running `cd packages/umi && uv run pytest` picks that up directly.

Honest state of the suite, as measured on this checkout:

- 155 tests are collected, all under `packages/umi/tests`. `packages/diffusion_policy` has no tests.
- **Known issue:** the suite **hangs indefinitely** in `packages/umi/tests/services/test_aruco_detection.py:149` (`test_execute_all_videos_processed`) and again in the test after it. A plain `uv run pytest packages/umi/tests` — without the `--ignore` above — never terminates. See [Known issues](./known-issues.md).
- Excluding that file, the result is **41 failed, 100 passed**. The failures cluster in the calibration and dataset-planning services.
- `uv run ruff check src packages` reports 317 errors (203 auto-fixable). There is no `[tool.ruff]` configuration anywhere, so this is ruff's default rule set.

Treat green tests as an aspiration, not a gate, and do not assume a failure you see is yours.

---

**Next:** [Getting started](./getting-started.md) · [Repository map](./repository-map.md) · [Known issues](./known-issues.md)
