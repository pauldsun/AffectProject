"""Offline per-frame feature extraction for videos (e.g. DAiSEE clips). Run from project root:
   python src/extract_features.py data/DAiSEE/DataSet --out results/features --fps 10
Writes one CSV per video (FEATURE_COLUMNS + clip/frame/time) and a matching .npz with, per kept
frame, the raw 98-point landmarks and the 1280-d multitask backbone embedding (float16; NaN rows
where no face). --no-landmarks / --no-embeddings drop either. Existing outputs are skipped, so an interrupted
run (e.g. Colab disconnect) resumes where it stopped."""
import argparse, os, sys, time
from pathlib import Path
import cv2, numpy as np, pandas as pd
from tqdm import tqdm
from features import FeatureExtractor, FEATURE_COLUMNS, EMBED_DIM

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}

def find_videos(inputs):
    """Yield (video_path, output_stem) pairs; directory inputs keep their subfolder layout."""
    for inp in map(Path, inputs):
        if inp.is_dir():
            for v in sorted(p for p in inp.rglob("*")         # skip macOS ._ metadata files
                            if p.suffix.lower() in VIDEO_EXTS and not p.name.startswith("._")):
                yield v, v.relative_to(inp).with_suffix("")
        else:
            yield inp, Path(inp.stem)


def sampled_frames(cap, target_fps):
    """Yield (frame_idx, t_sec, frame) at ~target_fps. Time comes from frame index / native fps,
    since CAP_PROP_POS_MSEC is unreliable for some AVI files. target_fps <= 0 keeps every frame."""
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = 1.0 / target_fps if target_fps > 0 else 0.0
    next_t, i = 0.0, 0
    while True:
        ok, frame = cap.read()                  # decode every frame; only the kept ones hit the models
        if not ok:
            return
        t = i / native
        if t + 1e-6 >= next_t:
            yield i, t, frame
            next_t += step
        i += 1


def extract_video(ex, video, target_fps):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise IOError(f"cannot open {video}")
    rows, lms, embs = [], [], []
    for i, t, frame in sampled_frames(cap, target_fps):
        r = ex.process(frame)
        rows.append({"clip_id": video.stem, "frame_idx": i, "t_sec": round(t, 4), **r.features})
        lms.append(r.landmarks if r.landmarks is not None else np.full((98, 2), np.nan, np.float32))
        embs.append(r.embedding if r.embedding is not None else np.full(EMBED_DIM, np.nan, np.float32))
    cap.release()
    if not rows:
        raise IOError(f"no frames decoded from {video}")
    df = pd.DataFrame(rows, columns=["clip_id", "frame_idx", "t_sec"] + FEATURE_COLUMNS)
    return df, np.stack(lms).astype(np.float32), np.stack(embs).astype(np.float16)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="video files and/or directories (searched recursively)")
    ap.add_argument("--out", default="results/features", help="output directory")
    ap.add_argument("--fps", type=float, default=10, help="sampling rate; 0 = every frame")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"], help="mps = Apple GPU")
    ap.add_argument("--no-landmarks", action="store_true", help="don't save raw landmarks in the .npz")
    ap.add_argument("--no-embeddings", action="store_true", help="don't save backbone embeddings in the .npz")
    ap.add_argument("--overwrite", action="store_true", help="redo videos that already have output")
    args = ap.parse_args()

    out_root = Path(args.out).resolve()
    jobs = [(v, out_root / stem) for v, stem in find_videos(args.inputs)]
    todo = [(v, o) for v, o in jobs if args.overwrite or not o.with_suffix(".csv").exists()]
    print(f"{len(jobs)} videos found, {len(jobs) - len(todo)} already done, {len(todo)} to process")
    if not todo:
        return

    ex = FeatureExtractor(device=args.device)
    failed = []
    for video, out in tqdm(todo, unit="video"):
        try:
            t0 = time.perf_counter()
            df, lms, embs = extract_video(ex, video, args.fps)
            out.parent.mkdir(parents=True, exist_ok=True)
            # write to temp names then rename: a crash mid-write never leaves a file that looks "done"
            arrays = {"frame_idx": df["frame_idx"].to_numpy()}
            if not args.no_landmarks:
                arrays["landmarks"] = lms
            if not args.no_embeddings:
                arrays["embedding"] = embs
            if len(arrays) > 1:
                np.savez_compressed(out.with_suffix(".tmp.npz"), **arrays)
                os.replace(out.with_suffix(".tmp.npz"), out.with_suffix(".npz"))
            df.to_csv(out.with_suffix(".tmp.csv"), index=False)
            os.replace(out.with_suffix(".tmp.csv"), out.with_suffix(".csv"))
            tqdm.write(f"{video.name}: {len(df)} frames, face in {df.face_detected.mean():.0%}, "
                       f"{len(df) / (time.perf_counter() - t0):.1f} frames/s")
        except Exception as e:                 # one bad clip shouldn't kill a multi-hour run
            failed.append(video)
            tqdm.write(f"FAILED {video}: {e}")
    if failed:
        print(f"{len(failed)} failed:", *failed, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
