"""Continuity audit: is clip N+1 really the next 10 s of the same person as clip N?  Run from project root:
   python src/audit_continuity.py data/DAiSEE/DataSet --labels data/DAiSEE/Labels/AllLabels.csv --out results/audit
   (add --max-subjects 10 for a quick first look; per-subject results are cached, so reruns resume)

Idea: if clip B follows clip A with no gap, A's LAST frame and B's FIRST frame are 1/30 s apart, so they
should differ about as much as two neighbouring frames inside one clip. For each subject we measure how much
frames change inside clips as a function of time lag (1 frame, 1 s, 3 s, 9 s) and read the boundary
difference off that curve as an estimated gap in seconds. Frames are compared as small grayscale
thumbnails, standardized per image so slow webcam exposure drift doesn't count as change.
Verdicts: continuous (gap < 0.5 s), short_gap (explained by <= 9 s of this subject's own motion; the second
estimate is rough), discontinuous (a bigger change than any within-clip lag: a long gap, a new session/camera
setup, or wrong order). A very still subject makes gaps look small, so check the montages, not just counts.

Writes to --out:
  clips.csv    per clip: frames, fps, resolution, duration, file size/mtime, embedded creation_time
  pairs.csv    per adjacent pair: boundary distance, reverse-order distance, random same-subject distance,
               estimated gap, verdict, label changes (with --labels)
  summary.txt  the numbers that matter (also printed)
  montage_<verdict>.png  A's last frame | B's first frame for sample pairs: check the verdicts by eye
"""
import argparse, json, os, pickle, random, subprocess
from multiprocessing import Pool
from pathlib import Path
import cv2, numpy as np, pandas as pd
from tqdm import tqdm
from daisee import find_clips, load_labels, AFFECTS

THUMB = (96, 54)                 # 16:9 thumbnail used for all distances
LAGS_S = [1, 3, 9]               # within-clip lags (plus 1 frame) that calibrate "how much does this person move"
CONTINUOUS_S, SHORT_GAP_S = 0.5, 9.0


def tiny(frame):
    g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), THUMB, interpolation=cv2.INTER_AREA).astype(np.float32)
    return (g - g.mean()) / (g.std() + 1e-6)


def dist(a, b):
    return float(np.abs(a - b).mean())


def creation_time(path):
    """Container creation_time tag if the recorder wrote one (often absent); needs ffprobe."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json", path],
                             capture_output=True, text=True, timeout=30).stdout
        tags = json.loads(out).get("format", {}).get("tags", {})
        return next((v for k, v in tags.items() if "creation" in k.lower() or k.lower() == "date"), None)
    except Exception:
        return None


def scan(path):
    """Decode one clip; keep thumbnails of the first two frames, frames at each lag, and the last two."""
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    meta = dict(path=path, fps=fps, w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                header_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), size_mb=os.path.getsize(path) / 1e6,
                mtime=os.path.getmtime(path), creation_time=creation_time(path))
    want = {round(l * fps): l for l in LAGS_S}
    thumbs, prev, last, n = {}, None, None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n in (0, 1) or n in want:
            thumbs[n] = tiny(frame)
        prev, last = last, frame
        n += 1
    cap.release()
    meta.update(frames=n, duration_s=n / fps)
    if n < 2:
        return meta, None
    t = dict(first=thumbs[0], second=thumbs[1], last=tiny(last), last2=tiny(prev))
    t["lags"] = {l: dist(thumbs[0], thumbs[k]) for k, l in want.items() if k in thumbs}
    t["lags"][1 / fps] = (dist(t["first"], t["second"]) + dist(t["last2"], t["last"])) / 2
    return meta, t


def scan_subject(pool, clips, cache):
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    res = list(pool.imap(scan, clips.path.tolist()))
    cache.write_bytes(pickle.dumps(res))
    return res


def gap_seconds(d, curve):
    """Invert the subject's lag -> distance curve (forced monotone). 0 = looks like one frame step."""
    lags = np.array(sorted(curve))
    ds = np.maximum.accumulate([curve[l] for l in lags])
    if d <= ds[0]:
        return 0.0
    if d > ds[-1]:
        return np.inf
    return float(np.interp(d, ds, lags))


def verdict(gap):
    return "continuous" if gap < CONTINUOUS_S else ("short_gap" if gap <= SHORT_GAP_S else "discontinuous")


def last_frame(path):
    cap, last = cv2.VideoCapture(path), None
    while True:
        ok, f = cap.read()
        if not ok:
            return last
        last = f


def first_frame(path):
    ok, f = cv2.VideoCapture(path).read()
    return f if ok else None


def montage(pairs, out, n=8, seed=0):
    for v, grp in pairs.groupby("verdict"):
        rows = []
        for _, p in grp.sample(min(n, len(grp)), random_state=seed).iterrows():
            a, b = last_frame(p.path_a), first_frame(p.path_b)
            if a is None or b is None:
                continue
            row = np.hstack([cv2.resize(a, (384, 216)), cv2.resize(b, (384, 216))])
            txt = f"{p.clip_a} last | {p.clip_b} first   d={p.d_next:.2f} (1 frame {p.d_1frame:.2f})  gap~{p.gap_s:.1f}s"
            for th, c in ((4, (0, 0, 0)), (1, (255, 255, 255))):
                cv2.putText(row, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, th, cv2.LINE_AA)
            rows.append(row)
        if rows:
            cv2.imwrite(str(out / f"montage_{v}.png"), np.vstack(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="DAiSEE DataSet folder (contains Train/Validation/Test)")
    ap.add_argument("--labels", help="AllLabels.csv, to report how often labels change between adjacent clips")
    ap.add_argument("--out", default="results/audit")
    ap.add_argument("--max-subjects", type=int, help="only the first N subjects (quick look)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "scan").mkdir(parents=True, exist_ok=True)
    clips = find_clips(args.root)
    if clips.empty:
        raise SystemExit(f"no videos under {args.root}")
    subjects = clips.subject.unique()[: args.max_subjects]
    print(f"{len(clips)} clips, {clips.subject.nunique()} subjects; scanning {len(subjects)}")

    rng = random.Random(args.seed)
    clip_rows, pair_rows = [], []
    with Pool(args.workers) as pool:
        for s in tqdm(subjects, unit="subject"):
            sc = clips[clips.subject == s].reset_index(drop=True)
            res = scan_subject(pool, sc, out / "scan" / f"{s}.pkl")
            for (_, c), (meta, _) in zip(sc.iterrows(), res):
                clip_rows.append({**c.to_dict(), **{k: v for k, v in meta.items() if k != "path"}})
            ok = [(c, t) for (_, c), (_, t) in zip(sc.iterrows(), res) if t is not None]
            if len(ok) < 2:
                continue
            curve = pd.DataFrame([t["lags"] for _, t in ok]).median().to_dict()   # subject's typical motion
            for i in range(len(ok) - 1):
                (ca, ta), (cb, tb) = ok[i], ok[i + 1]
                others = [t for j, (_, t) in enumerate(ok) if j not in (i, i + 1)]
                d_next = dist(ta["last"], tb["first"])
                gap = gap_seconds(d_next, curve)
                pair_rows.append(dict(
                    subject=s, split_a=ca.split, split_b=cb.split, clip_a=ca.clip_id, clip_b=cb.clip_id,
                    path_a=ca.path, path_b=cb.path,
                    id_step=(int(cb.clip_id) - int(ca.clip_id)) if ca.clip_id.isdigit() and cb.clip_id.isdigit() else None,
                    d_next=d_next,                                         # A last vs B first
                    d_rev=dist(tb["last"], ta["first"]),                   # B last vs A first: small = order reversed
                    d_rand=np.median([dist(ta["last"], t["first"]) for t in rng.sample(others, min(3, len(others)))])
                    if others else np.nan,                                 # same person, unrelated clip
                    d_1frame=min(curve.values()), gap_s=gap, verdict=verdict(gap)))

    clip_df, pairs = pd.DataFrame(clip_rows), pd.DataFrame(pair_rows)
    if args.labels and not pairs.empty:
        lab = load_labels(args.labels).set_index("clip_id")
        for side in "ab":
            pairs = pairs.join(lab.add_suffix(f"_{side}"), on=f"clip_{side}")
        for a in AFFECTS:                      # NaN (not "changed") when either clip has no label
            both = pairs[f"{a}_a"].notna() & pairs[f"{a}_b"].notna()
            pairs[f"changed_{a}"] = (pairs[f"{a}_a"] != pairs[f"{a}_b"]).astype(float).where(both)
        pairs["changed_any"] = pairs[[f"changed_{a}" for a in AFFECTS]].max(axis=1)
    clip_df.to_csv(out / "clips.csv", index=False)
    pairs.to_csv(out / "pairs.csv", index=False)

    lines = summarize(clip_df, pairs, args.labels is not None)
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not pairs.empty:
        montage(pairs, out, seed=args.seed)
        print(f"\nmontages + csvs in {out}/  (look at montage_*.png to check the verdicts by eye)")


def summarize(clips, pairs, has_labels):
    L = ["== clip format =="]
    for col in ["w", "h", "fps"]:
        L.append(f"{col}: " + ", ".join(f"{k} ({v})" for k, v in clips[col].round(2).value_counts().head(5).items()))
    L.append(f"frames: median {clips.frames.median():.0f}, min {clips.frames.min()}, max {clips.frames.max()};  "
             f"duration median {clips.duration_s.median():.2f} s;  size median {clips.size_mb.median():.1f} MB, "
             f"total {clips.size_mb.sum() / 1e3:.1f} GB (scanned)")
    L.append(f"header frame count != decoded frames: {(clips.header_frames != clips.frames).sum()} clips;  "
             f"creation_time present: {clips.creation_time.notna().sum()} clips")
    multi = clips.groupby("subject").split.nunique()
    L.append(f"subjects in >1 split: {(multi > 1).sum()} (should be 0 for subject-disjoint splits)")
    if pairs.empty:
        return L
    L.append(f"\n== adjacent pairs ({len(pairs)}) ==")
    for v, n in pairs.verdict.value_counts().items():
        L.append(f"{v:14s} {n:6d}  ({n / len(pairs):.0%})")
    L.append(f"median distance: 1 frame {pairs.d_1frame.median():.3f} | boundary {pairs.d_next.median():.3f} | "
             f"reversed order {pairs.d_rev.median():.3f} | random same-subject clip {pairs.d_rand.median():.3f}")
    L.append(f"pairs where reversed order fits better than forward: {(pairs.d_rev < pairs.d_next).mean():.0%}")
    if pairs.id_step.notna().any():
        L.append("clip-id step between neighbours: " + ", ".join(
            f"{int(k)} ({v})" for k, v in pairs.id_step.value_counts().head(6).items()))
        L.append("continuous rate by id step: " + ", ".join(
            f"{int(k)}: {g.eq('continuous').mean():.0%}" for k, g in pairs.groupby("id_step").verdict
            if len(g) >= 5))
    # chains of continuous clips = how much usable history the forecasting windows get
    chain_lens = []
    for _, g in pairs.groupby("subject"):
        run = 1
        for c in g.verdict.eq("continuous"):
            if c:
                run += 1
            else:
                chain_lens.append(run); run = 1
        chain_lens.append(run)
    cl = pd.Series(chain_lens)
    L.append(f"continuous chains: {len(cl)}, length in clips median {cl.median():.0f}, max {cl.max()};  "
             f"clips with >=1 continuous predecessor: {(cl - 1).sum()}")
    if has_labels and "changed_any" in pairs:
        L.append("\n== label change between adjacent clips (persistence baseline preview) ==")
        for name, g in [("all pairs", pairs), ("continuous pairs", pairs[pairs.verdict == "continuous"])]:
            if len(g):
                L.append(f"{name:16s} n={len(g):5d}  any affect changed {g.changed_any.mean():.0%}  " + "  ".join(
                    f"{a} {g[f'changed_{a}'].mean():.0%}" for a in AFFECTS))
    return L


if __name__ == "__main__":
    main()
