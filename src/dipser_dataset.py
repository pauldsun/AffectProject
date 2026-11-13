"""DIPSER: per-frame aligned tables -> one row per student-second, ready for detection and forecasting models.

Reads aligned/<rec>/frames.csv.gz (dipser_align.py), features/<rec>/images.npz (embeddings) and the self-report
labels from videos/<rec>/extras.tar.gz. Writes
  <out>/seconds.csv.gz   one row per second of every recording (columns below)
  <out>/seconds_emb.npz  emb (rows, 1280) float16: mean backbone embedding over the second's face frames,
                         in the same row order (reduce it, e.g. PCA, inside each training fold, never here)

Time. Second k of a recording covers t in [k, k+1) from its first frame. A model "decides" at the END of the
second (t = k+1), having seen all of its frames and nothing later. Seconds with no frames (camera gaps up to 5 s)
are kept, with n_frames = 0 and empty features, so the sequence stays evenly spaced.

Features per second (from our extractor only, unless the name starts with data_):
  n_frames, face_frac                    how much was seen
  <f>_mean for every feature             over the frames with a face
  <f>_sd   for the moving ones           head angles, gaze, eye/mouth opening, brow raise
  eye_open_min                           blinks / closing eyes
  head_speed                             median angular speed of the head, deg/s, from the true frame times
  data_hand_left, data_hand_right        share of frames where the DATASET's detector sees a hand (posture
                                         stand-in until we run our own pose; keep separate in ablations)
  z_<col>                                the same, relative to this student's own past: (x - mean) / sd over
                                         the earlier seconds of the recording only (causal); empty for the
                                         first --warmup seconds. Removes "who" so a model must learn "when".

Targets (from the self-report label change points, at the decision time t = k+1):
  bored_now          1 / 0                       label in force at the decision time
  time_to_bored      s until the next switch to boredom (empty if none before the recording ends); only
                     defined while not bored
  since_bored        s since the student last stopped being bored (empty if never bored so far)
  onset_<H>s         1 if not bored now and bored within H s, 0 if not bored now and still not bored H s later,
                     empty if bored now or the recording ends first (unknown)

  python src/dipser_dataset.py --data data/dipser --out data/dipser/dataset --include experiment_03
"""
import argparse, json, sys, tarfile
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dipser
from features import FEATURE_COLUMNS

HORIZONS = (5, 10, 15, 30)
SKIP = {"face_detected", "n_faces"}                       # covered by face_frac
MOVING = ["head_pitch", "head_yaw", "head_roll", "gaze_yaw", "gaze_pitch", "eye_open", "mouth_open", "brow_raise"]


def self_events(tar_path):
    """Self-report label entries only (skips the per-frame metadata, which is most of the archive)."""
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf:
            if m.isfile() and "labels" in Path(m.name).parts and Path(m.name).stem.startswith("self"):
                return pd.DataFrame(dipser.label_rows(json.load(tf.extractfile(m)), "self"))
    return pd.DataFrame()


def per_second(fr):
    """Frame rows (one recording, aligned) -> one row per second, features only."""
    face = fr.face_detected == 1
    feats = [c for c in FEATURE_COLUMNS if c not in SKIP]
    k = fr.t.astype(int)
    n_sec = int(fr.t.iloc[-1]) + 1
    out = pd.DataFrame(index=pd.RangeIndex(n_sec, name="sec"))
    out["n_frames"] = fr.groupby(k).size().reindex(out.index, fill_value=0)
    out["face_frac"] = face.groupby(k).mean().reindex(out.index)
    f = fr[face]
    kf = k[face]
    m = f[feats].groupby(kf).mean().add_suffix("_mean")
    s = f[MOVING].groupby(kf).std().add_suffix("_sd")
    out = out.join(m).join(s)
    out["eye_open_min"] = f.eye_open.groupby(kf).min()
    # head angular speed between consecutive face frames, using the true time between them
    d = np.hypot(f.head_pitch.diff(), f.head_yaw.diff()) / f.t.diff()
    d[f.frame_idx.diff() != 1] = np.nan                   # only across directly consecutive frames
    out["head_speed"] = d.groupby(kf).median()
    for side in ("left", "right"):
        col = f"meta_hand_{side}"
        if col in fr:
            out[f"data_hand_{side}"] = fr[col].groupby(k).mean().reindex(out.index)
    return out


def causal_z(df, cols, warmup):
    """(x - mean of earlier seconds) / sd of earlier seconds; empty until `warmup` seconds with values."""
    past = df[cols].shift(1)
    mean = past.expanding(min_periods=warmup).mean()
    sd = past.expanding(min_periods=warmup).std()
    return ((df[cols] - mean) / sd.where(sd > 1e-9)).add_prefix("z_")


def targets(ev, clock0, n_sec):
    """bored_now / time_to_bored / since_bored / onset_<H>s at each decision time clock0 + k + 1."""
    t_dec = clock0 + np.arange(n_sec) + 1.0
    t_end = clock0 + n_sec                                # last moment we have video for
    out = pd.DataFrame(index=pd.RangeIndex(n_sec, name="sec"))
    if ev.empty:
        return out
    emo = ev.dropna(subset=["emotion"]).sort_values(["t", "order"], kind="stable")
    now = dipser.values_at(ev, t_dec)["emo_self"]
    out["bored_now"] = np.where(np.isnan(now), np.nan, (now == dipser.BOREDOM).astype(float))
    # switches into / out of boredom, as times
    b = (emo.emotion.values == dipser.BOREDOM)
    tt = emo.t.values
    starts = tt[1:][b[1:] & ~b[:-1]] if len(b) > 1 else np.array([])
    if len(b) and b[0]:
        starts = np.r_[tt[0], starts]
    ends = tt[1:][~b[1:] & b[:-1]] if len(b) > 1 else np.array([])
    i = np.searchsorted(starts, t_dec, side="right")      # first start after the decision time
    nxt = starts[np.minimum(i, len(starts) - 1)] if len(starts) else np.full(n_sec, np.nan)
    nxt = np.where(i < len(starts), nxt, np.nan)
    ttb = nxt - t_dec
    ttb[(out.bored_now != 0).values | (nxt > t_end)] = np.nan
    out["time_to_bored"] = ttb
    j = np.searchsorted(ends, t_dec, side="right") - 1    # last end at or before the decision time
    out["since_bored"] = (np.where(j >= 0, t_dec - ends[np.maximum(j, 0)], np.nan) if len(ends)
                          else np.full(n_sec, np.nan))
    for H in HORIZONS:
        y = np.full(n_sec, np.nan)
        free = (out.bored_now == 0).values
        y[free & (ttb <= H)] = 1
        y[free & ~(ttb <= H) & (t_dec + H <= t_end)] = 0  # stayed not bored for the full H s we can see
        out[f"onset_{H}s"] = y
    return out


def build(rec, root, warmup):
    fr = pd.read_csv(root / "aligned" / rec / "frames.csv.gz")
    g, e, s = map(int, dipser.REC_RE.search(rec.as_posix()).groups())
    sec = per_second(fr)
    zcols = [c for c in sec if c not in ("n_frames", "face_frac")]   # face_frac is ~always 1: no spread
    sec = sec.join(causal_z(sec, zcols, warmup))
    ev = self_events(root / "videos" / rec / "extras.tar.gz")
    sec = sec.join(targets(ev, fr.clock.iloc[0], len(sec)))
    sec.insert(0, "group", g), sec.insert(1, "experiment", e), sec.insert(2, "subject", s)
    sec.insert(3, "student", f"g{g}s{s}")                 # the unit for student-disjoint splits

    z = np.load(root / "features" / rec / "images.npz")
    emb = pd.DataFrame(z["embedding"].astype(np.float32), index=z["frame_idx"])
    keep = fr.set_index("frame_idx").reindex(emb.index)
    ok = (keep.face_detected == 1).values
    E = emb[ok].groupby(keep.t[ok].astype(int).values).mean().reindex(sec.index)
    return sec.reset_index(), E.to_numpy(np.float16)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/dipser", help="folder with videos/, features/, aligned/")
    ap.add_argument("--out", default="data/dipser/dataset")
    ap.add_argument("--include", default="", help="only recordings whose path contains this, e.g. experiment_03")
    ap.add_argument("--warmup", type=int, default=20, help="seconds of a student's past before z_ columns start")
    a = ap.parse_args()

    root, out = Path(a.data), Path(a.out)
    recs = sorted(p.parent.relative_to(root / "aligned") for p in (root / "aligned").glob("**/frames.csv.gz")
                  if a.include in p.as_posix())
    print(f"{len(recs)} recordings")
    rows, embs = [], []
    for i, rec in enumerate(recs, 1):
        try:
            df, E = build(rec, root, a.warmup)
        except Exception as ex:
            print(f"[{i}/{len(recs)}] {rec}: FAILED {type(ex).__name__}: {ex}")
            continue
        rows.append(df), embs.append(E)
        print(f"[{i}/{len(recs)}] {rec}: {len(df)} s, bored {df.bored_now.mean():.0%}, "
              f"onsets within 10 s: {int(df.onset_10s.sum())} s")
    if not rows:
        return
    d = pd.concat(rows, ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "seconds.csv.gz", index=False, float_format="%.6g")
    np.savez_compressed(out / "seconds_emb.npz", emb=np.concatenate(embs))

    print(f"\n== {len(d)} student-seconds from {d.student.nunique()} students -> {out}")
    print(f"seconds with no frames: {(d.n_frames == 0).mean():.1%} | with a face in every frame: "
          f"{(d.face_frac == 1).mean():.1%} | bored now: {d.bored_now.mean():.1%}")
    print(f"z_ columns start after {a.warmup} s: {d.z_head_pitch_mean.notna().mean():.0%} of seconds have them")
    print("forecasting samples (not bored now, outcome known):")
    for H in HORIZONS:
        y = d[f"onset_{H}s"].dropna()
        print(f"  H = {H:2d} s: {len(y):6d} samples, {y.mean():.1%} followed by boredom, "
              f"{d.loc[y.index, 'student'].nunique()} students")


if __name__ == "__main__":
    main()
