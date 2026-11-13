"""DIPSER: put our per-frame features, the dataset's per-frame metadata and the labels on one clock.

For every recording fetched by dipser_fetch.py (videos/<rec>/done.json) it reads
  videos/<rec>/images.frames.csv   video frame -> PNG name -> clock time (s since midnight)
  videos/<rec>/extras.tar.gz       metadata/<png name>.json (the dataset's own face analysis) + labels/*.json
  features/<rec>/images.csv        our features from extract_features.py --fps 0 (optional: skipped if missing)
and writes
  aligned/<rec>/frames.csv.gz      one row per video frame: ids, clock, t, dt, our features, meta_* columns,
                                   att_<rater> / emo_<rater> label in force at that frame, bored_self
  aligned/<rec>/meta_body.npz      body_pose (frames, 33, 4: x, y, z, visibility) from the metadata
  aligned/summary.csv              one row per recording: frame timing, face coverage, label coverage,
                                   correlation of our head pose with the dataset's

Frame time comes from frames.csv, never from the mp4: the mp4 is written at a nominal 10 fps but the camera ran at
5.7-9 fps with gaps. Our features join on frame_idx == video_frame (extract_features.py --fps 0 keeps every frame;
its t_sec column assumes 10 fps and is dropped here). The 1280-d embeddings stay in features/<rec>/images.npz, in
the same row order as features/<rec>/images.csv.

Metadata fields NOT kept: age, gender, race (predicted demographics; not used as features), the 478-point face mesh
(large; our own landmarks cover the face). Emotion probabilities are converted from percent to 0-1.

  python src/dipser_align.py --videos data/dipser/videos --features data/dipser/features --out data/dipser/aligned
"""
import argparse, json, sys, tarfile
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dipser

META_EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def read_extras(tar_path):
    """-> ({png stem: metadata dict}, {rater: label entries})."""
    meta, labels = {}, {}
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf:
            if not (m.isfile() and m.name.endswith(".json")):
                continue
            parts = Path(m.name).parts
            if "metadata" in parts:
                meta[Path(m.name).stem] = json.load(tf.extractfile(m))
            elif "labels" in parts:
                labels[dipser.rater_name(Path(m.name).stem)] = json.load(tf.extractfile(m))
    return meta, labels


def meta_row(d):
    """Flatten the fields we keep from one frame's metadata (NaN where missing)."""
    face = (d or {}).get("person", {}).get("face") or {}
    body = (d or {}).get("person", {}).get("body") or {}
    box = face.get("bounding_box") or {}
    if isinstance(box, (list, tuple)):              # a few frames store it as [x0, y0, x1, y1]
        box = dict(zip(("x0", "y0", "x1", "y1"), box)) if len(box) == 4 else {}
    pose = (face.get("headpose") or {}).get("pose") or {}
    emo = (face.get("emotion") or {}).get("probability_emotion") or {}
    hands = body.get("hand_pose") or {}
    r = {"meta_face": float(bool(box))}
    for k in ("x0", "y0", "x1", "y1"):
        r[f"meta_face_{k}"] = box.get(k, np.nan)
    for k in ("pitch", "yaw", "roll"):
        r[f"meta_head_{k}"] = pose.get(k, np.nan)
    for k in META_EMOTIONS:
        r[f"meta_emo_{k}"] = emo[k] / 100 if k in emo else np.nan
    r["meta_hand_left"] = float(bool(hands.get("left_hand")))
    r["meta_hand_right"] = float(bool(hands.get("right_hand")))
    return r


def body_pose(d):
    pts = ((d or {}).get("person", {}).get("body") or {}).get("body_pose") or []
    a = np.full((33, 4), np.nan, np.float32)
    for i, p in enumerate(pts[:33]):
        a[i] = [p.get("x", np.nan), p.get("y", np.nan), p.get("z", np.nan), p.get("visibility", np.nan)]
    return a


def align(rec_dir, rel, feat_root):
    g, e, s = map(int, dipser.REC_RE.search(rel.as_posix()).groups())
    fr = pd.read_csv(rec_dir / "images.frames.csv")
    meta, labels = read_extras(rec_dir / "extras.tar.gz")
    stems = fr.file.map(lambda f: Path(f).stem)

    df = pd.DataFrame({"group": g, "experiment": e, "subject": s, "frame_idx": fr.video_frame,
                       "clock": fr.time_s})
    df["t"] = df.clock - df.clock.iloc[0]
    df["dt"] = df.clock.diff()

    feat_csv = feat_root / rel / "images.csv" if feat_root else None
    n_feat = None
    if feat_csv and feat_csv.exists():
        ft = pd.read_csv(feat_csv).drop(columns=["clip_id", "t_sec"], errors="ignore")
        n_feat = len(ft)
        df = df.merge(ft, on="frame_idx", how="left")

    df = pd.concat([df, pd.DataFrame([meta_row(meta.get(k)) for k in stems], index=df.index)], axis=1)

    ev = pd.DataFrame([row for r, entries in labels.items() for row in dipser.label_rows(entries, r)])
    if len(ev):
        for c, v in dipser.values_at(ev, df.clock.values).items():
            df[c] = v
    if "emo_self" in df:
        df["bored_self"] = np.where(df.emo_self.isna(), np.nan, (df.emo_self == dipser.BOREDOM).astype(float))

    poses = np.stack([body_pose(meta.get(k)) for k in stems]).astype(np.float16)

    gaps = df.dt.dropna()
    summ = dict(group=g, experiment=e, subject=s, frames=len(df), seconds=round(df.t.iloc[-1], 1),
                fps_median=round(1 / gaps.median(), 2), gap_max_s=round(gaps.max(), 2),
                gaps_over_0_5s=int((gaps > 0.5).sum()), meta_frames=sum(k in meta for k in stems),
                meta_face=round(df.meta_face.mean(), 3), raters=" ".join(sorted(labels)))
    if n_feat is not None:
        summ.update(feat_rows=n_feat, face_ours=round(df.face_detected.mean(), 3))
        for k in ("pitch", "yaw", "roll"):
            summ[f"r_{k}"] = round(df[f"head_{k}"].corr(df[f"meta_head_{k}"]), 2)
    if ev is not None and len(ev):
        summ["labels_from_s"] = round(ev.t.min() - df.clock.iloc[0], 1)     # first label relative to first frame
    if "bored_self" in df:
        summ["self_covered"] = round(df.bored_self.notna().mean(), 3)
        summ["bored_self"] = round(df.bored_self.mean(), 3)
    return df, poses, summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, help="videos/ folder written by dipser_fetch.py")
    ap.add_argument("--features", help="features/ folder written by extract_features.py (optional)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--include", default="", help="only recordings whose path contains this, e.g. experiment_03")
    a = ap.parse_args()

    videos, out = Path(a.videos), Path(a.out)
    feat_root = Path(a.features) if a.features else None
    recs = sorted(p.parent for p in videos.glob("**/done.json") if a.include in p.as_posix())
    print(f"{len(recs)} recordings")
    rows = []
    for i, rec in enumerate(recs, 1):
        rel = rec.relative_to(videos)
        try:
            df, poses, summ = align(rec, rel, feat_root)
        except Exception as ex:                       # keep going; the summary says which ones failed
            print(f"[{i}/{len(recs)}] {rel}: FAILED {type(ex).__name__}: {ex}")
            continue
        dst = out / rel
        dst.mkdir(parents=True, exist_ok=True)
        df.to_csv(dst / "frames.csv.gz", index=False, float_format="%.12g")   # clock needs µs: 68268.039858
        np.savez_compressed(dst / "meta_body.npz", body_pose=poses)
        rows.append(summ)
        extra = f", head pose r {summ['r_pitch']}/{summ['r_yaw']}/{summ['r_roll']}" if "r_pitch" in summ else ""
        print(f"[{i}/{len(recs)}] {rel}: {summ['frames']} frames, {summ['seconds']} s, "
              f"{summ.get('self_covered', 0):.0%} with a self label{extra}")
    if not rows:
        return
    s = pd.DataFrame(rows)
    s.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"\n== {len(s)} recordings -> {out}/summary.csv")
    print("frame timing: median fps", s.fps_median.median(), "| recordings with a gap > 0.5 s:",
          int((s.gaps_over_0_5s > 0).sum()), "| largest gap", s.gap_max_s.max(), "s")
    print("first self label relative to first frame (s):", s.labels_from_s.describe()[["min", "50%", "max"]].round(1).to_dict()
          if "labels_from_s" in s else "-")
    print("frames with a self label:", f"{s.self_covered.mean():.1%}" if "self_covered" in s else "-",
          "| bored (self):", f"{(s.bored_self * s.frames).sum() / s.frames.sum():.1%}" if "bored_self" in s else "-")
    print("dataset metadata: face found in", f"{(s.meta_face * s.frames).sum() / s.frames.sum():.1%}", "of frames")
    if "r_pitch" in s:
        f = s[s.feat_rows.notna()]
        print(f"our features found for {len(f)} of {len(s)} recordings; rows == frames in",
              int((f.feat_rows == f.frames).sum()), "| ours: face found in",
              f"{(f.face_ours * f.frames).sum() / f.frames.sum():.1%}", "of frames")
        print("head pose, ours vs dataset, correlation per recording (median [min, max]):")
        for k in ("pitch", "yaw", "roll"):
            c = s[f"r_{k}"]
            print(f"  {k:5s} {c.median():.2f} [{c.min():.2f}, {c.max():.2f}]")


if __name__ == "__main__":
    main()
