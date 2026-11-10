# AER1515: Affect Perception Project for AI Tutors

Using OpenFace3.0 for feature extraction, Perspective-n-Point (PnP) for 2D to 3D mapping, Kalman smoothing, and LSTM classification based on the DAiSEE dataset.

## Environment
- macOS (Apple Silicon, M1), python 3.11 venv managed by [uv](https://docs.astral.sh/uv/)
- OpenFace3.0 @ commit https://github.com/CMU-MultiComp-Lab/OpenFace-3.0.git
- All setup steps below are macOS specific (`sed -i ''`, BSD `stat`)

## Repo Layout
- `src/features.py` - shared per-frame features: OpenFace 3.0 (emotion, AUs, gaze) + PnP head pose + landmark geometry
- `src/extract_features.py` - offline extraction over videos/folders → one CSV (+ .npz of landmarks and 1280-d backbone embeddings) per clip, resumable
- `src/audit_continuity.py` - checks whether consecutive DAiSEE clips are back-to-back footage; label change rates
- `src/sample_clips.py` - copies a small reproducible subset of raw clips (full subjects + consecutive runs) for viewing
- `src/daisee.py` - DAiSEE folder layout + label loading helpers
- `src/live_demo.py` - webcam overlay of the same features; `--record` logs them to CSV
- `src/visualize_frame.py` - draw a `openface detect` CSV over its image
- `scripts/` - setup helpers (weight download, macOS patch)
- `OpenFace-3.0` - cloned repo **gitignored**
- `.venv/`, `data/`, `results/` - gitignored (env, DAiSEE, outputs)

## Setup from scratch
```bash 
## 0. Prereqs: git
## Install uv:
curl -LsSf https://astral.sh/uv/install.sh | sh #restart terminal after

## 1. Get the code
git clone https://github.com/pauldsun/AffectProject.git AffectProject
cd AffectProject
git clone https://github.com/CMU-MultiComp-Lab/OpenFace-3.0.git

## 2. Environment
uv python install 3.11
uv venv --python 3.11 --seed .venv
source .venv/bin/activate  # for every session
python --version           # sanity check (expect 3.11.x)

## 3. Dependencies
uv pip install -r OpenFace-3.0/requirements.txt
uv pip install openface-test

## 4. Model weights + macOS patch
./scripts/download_weights.sh      # ~460 MB, verifies sizes
./scripts/patch_openface.sh        # fixes hardcoded /work path (see Problems)

## 5. Verify (run from inside OpenFace-3.0 due to weights)
cd OpenFace-3.0
openface detect images/0.jpg --output-dir ../results --device cpu
## results/face_analysis_0.csv appears with landmarks/gaze/AU/emotion columns
```

## Problems:
1. `openface download` skips if `weights/` already exists, but `weights/` always comes with a fresh clone of OpenFace-3.0. Use `scripts/download_weights.sh` instead.
2. The pip package hardcodes the author's Docker path (`/work/...`) and crashes on macOS with `[Errno 30] Read-only file system`. To fix this, use 'scripts/patch_openface.sh'. **after every venv rebuild**
3. The CLI defaults to `--device cuda`. On Mac, always use `--device cpu`.
4. The OpenFace-3.0 `demo.py` and `demo2.py` rely on empty directories, thus they are unusable.
5. `detect-video` passes ndarray frames into a path-only API; use your own loop.

## Usage (from project root)
```bash
python src/live_demo.py --device mps                                   # webcam, q to quit
python src/extract_features.py data/DAiSEE/DataSet --out results/features --fps 10 --device mps
python src/audit_continuity.py data/DAiSEE/DataSet --labels data/DAiSEE/Labels/AllLabels.csv --out results/audit
python src/sample_clips.py data/DAiSEE/DataSet --labels data/DAiSEE/Labels/AllLabels.csv --out data/samples
```
`--device`: `mps` (Apple GPU; landmarks stay on CPU), `cuda` (Colab/lab GPU), or `cpu`.
Column meanings and caveats are documented at the top of `src/features.py`.

On Colab: open `colab.ipynb` from GitHub (File > Open notebook > GitHub). It clones this repo, downloads DAiSEE
and runs the same scripts with `!python ...`, writing results to Google Drive. It also does the Linux version
of the setup above (OpenFace install, weights, patch).

## Status Log:
- 2026-10-17: Environment Setup + OpenFace working. Next: visualization script and feature extraction script.
- 2026-11-02: Shared feature module + offline extractor + live demo rewrite. AU output order looks like
  DISFA-8 (AU1,2,4,6,9,12,25,26), not `eval_au` from train_mix_au_dev.py; verify on webcam.
- 2026-11-06: Webcam: smile -> au_5 and pitch sign confirmed; frown (au_2) unconfirmed. Fixed STAR crop
  (OpenFace used the box's short side, cutting off the chin): jawline now on the chin, head pitch shifted
  by up to ~10 deg on frontal faces. Re-extract anything produced before this fix.
- 2026-11-08: .npz now also stores the multitask backbone embedding (1280-d, float16, ~2.7 GB for DAiSEE).
  Added continuity audit (tested on clips cut from test.avi: back-to-back, 4 s gap, 8 s gap, out of order all
  classified correctly) and raw-clip sampler. Next: download DAiSEE in Colab, run the audit before extraction.
- 2026-11-09: Added `colab.ipynb` (session A on CPU: download + audit + samples; session B on GPU: extraction).
  Colab parts untested so far (Colab was down).
- 2026-11-09: DAiSEE unpacked on Colab: 9,068 clips (8,232 .avi + 836 .mp4), 8,925 label rows (143 clips unlabeled).
  1 clip has a bad CRC in the official zip (Train/210055/2100552061), set aside by the notebook. Dataset README:
  clip-number holes come from data cleaning (removed clips), so id gaps likely mean missing time.

## Citations:
```bibtex
@article{hu2025openface,
  title={OpenFace 3.0: A Lightweight Multitask System for Comprehensive Facial Behavior Analysis},
  author={Hu, Jiewen and Mathur, Leena and Liang, Paul Pu and Morency, Louis-Philippe},
  journal={arXiv preprint arXiv:2506.02891},
  year={2025}
}
```
