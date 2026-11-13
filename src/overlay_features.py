"""Draw extracted features back onto a DIPSER recording -> annotated mp4, to check by eye that they look right.

No models are run: everything comes from the files extract_features.py wrote.
  features/<rec>/images.csv   per-frame features        features/<rec>/images.npz   98 landmarks per frame
  videos/<rec>/images.mp4      the frames                videos/<rec>/images.frames.csv   true clock per frame
  aligned/<rec>/frames.csv.gz  optional (dipser_align.py): self-reported boredom + the dataset's head pose

Per frame: landmarks, head-pose axes (PnP recomputed from the saved landmarks, red/green/blue = face
x/y/z), top emotion, head pose (ours, and the dataset's if aligned), gaze, face geometry and AU bars. Underneath:
a strip over the whole recording (self-reported boredom in red, our head pitch and yaw, eye opening) with a
playhead. Frames are repeated to follow the true clock, so playback runs at real speed and the camera's gaps
show as freezes (the clock in the corner keeps the true time).

  python src/overlay_features.py DIPSEER/group_03/experiment_03/subject_14 --data data/dipser
  python src/overlay_features.py DIPSEER/group_03/experiment_03/subject_14 --data data/dipser --start 60 --seconds 30
"""
import argparse, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import EMOTIONS, AU_LABELS, NO_DISTORTION, head_pose

OUT_FPS = 10
STRIP_H = 150


def text(img, lines, x=8, y=18, color=(255, 255, 255), scale=0.42, step=17):
    for k, s in enumerate(lines):
        org = (x, y + step * k)
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def au_bars(img, f, y0, bar_w=80):
    for i in range(8):
        y = y0 + 17 * i
        s = float(np.clip(f[f"au_{i}"], 0, 1)) if pd.notna(f[f"au_{i}"]) else 0.0
        cv2.rectangle(img, (8, y - 10), (8 + bar_w, y + 2), (0, 0, 0), -1)
        cv2.rectangle(img, (8, y - 10), (8 + int(bar_w * s), y + 2), (0, 165, 255) if s > 0.5 else (200, 200, 200), -1)
        text(img, [f"au_{i} {s:.2f} {AU_LABELS[i]}?"], x=14 + bar_w, y=y)


def norm(v):
    lo, hi = np.nanpercentile(v, [2, 98]) if np.isfinite(v).any() else (0, 1)
    return np.clip((v - lo) / (hi - lo + 1e-9), 0, 1)


def make_strip(df, w):
    """Whole-recording timeline: one row per signal, each scaled to its own 2-98 percentile range."""
    s = np.full((STRIP_H, w, 3), 30, np.uint8)
    t = df.t.values
    X = lambda tt: (8 + (tt / t[-1]) * (w - 16)).astype(int)
    rows = [("bored (self)", None, (60, 60, 230)), ("head pitch", "head_pitch", (230, 160, 60)),
            ("head yaw", "head_yaw", (60, 180, 255)), ("eye open", "eye_open", (120, 220, 120))]
    band = (STRIP_H - 20) // len(rows)
    for k, (name, col, color) in enumerate(rows):
        y0 = 4 + k * band
        if col is None:
            if "bored_self" in df:
                for i in np.flatnonzero(df.bored_self.values == 1):
                    j = min(i + 1, len(t) - 1)
                    cv2.rectangle(s, (X(t[i:i + 1])[0], y0 + 4), (X(t[j:j + 1])[0], y0 + band - 4), color, -1)
            else:
                text(s, ["(run dipser_align.py for labels)"], x=110, y=y0 + band // 2 + 4, color=(160, 160, 160))
        else:
            v = norm(df[col].values.astype(float))
            pts = np.stack([X(t), (y0 + band - 4 - v * (band - 8)).astype(int)], 1)
            ok = np.isfinite(df[col].values)
            for a, b in zip(pts[:-1][ok[:-1] & ok[1:]], pts[1:][ok[:-1] & ok[1:]]):
                cv2.line(s, tuple(map(int, a)), tuple(map(int, b)), color, 1, cv2.LINE_AA)
        text(s, [name], x=8, y=y0 + 14, color=color)
    for m in range(0, int(t[-1]) + 1, 60):
        x = X(np.array([m]))[0]
        cv2.line(s, (x, STRIP_H - 16), (x, STRIP_H - 12), (200, 200, 200), 1)
        text(s, [f"{m // 60}:00"], x=max(0, x - 12), y=STRIP_H - 2, color=(200, 200, 200))
    return s, X


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rec", help="recording path under videos/, e.g. DIPSEER/group_03/experiment_03/subject_14")
    ap.add_argument("--data", default="data/dipser", help="folder holding videos/, features/ and aligned/")
    ap.add_argument("--out", help="output mp4 (default: <data>/overlays/<rec>.mp4)")
    ap.add_argument("--start", type=float, default=0, help="start time in s from the first frame")
    ap.add_argument("--seconds", type=float, default=0, help="length in s; 0 = to the end")
    a = ap.parse_args()

    root, rec = Path(a.data), Path(a.rec)
    vid, feat = root / "videos" / rec, root / "features" / rec
    fr = pd.read_csv(vid / "images.frames.csv")
    ft = pd.read_csv(feat / "images.csv").drop(columns=["clip_id", "t_sec"], errors="ignore")
    z = np.load(feat / "images.npz")
    lms = dict(zip(z["frame_idx"].tolist(), z["landmarks"]))
    df = pd.DataFrame({"frame_idx": fr.video_frame, "t": fr.time_s - fr.time_s.iloc[0], "clock": fr.time_s})
    df = df.merge(ft, on="frame_idx", how="left")
    al = root / "aligned" / rec / "frames.csv.gz"
    if al.exists():
        keep = ["frame_idx", "bored_self", "meta_head_pitch", "meta_head_yaw", "meta_head_roll", "emo_self"]
        alc = pd.read_csv(al, usecols=lambda c: c in keep)
        df = df.merge(alc, on="frame_idx", how="left")
    else:
        print(f"note: {al} not found, so no labels or dataset head pose (run dipser_align.py first)")

    cap = cv2.VideoCapture(str(vid / "images.mp4"))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    strip, X = make_strip(df, w)
    out = Path(a.out) if a.out else root / "overlays" / rec.with_suffix(".mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), OUT_FPS, (w, h + STRIP_H))

    t_end = a.start + a.seconds if a.seconds > 0 else np.inf
    by_idx = df.set_index("frame_idx")
    written, i = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i not in by_idx.index:
            i += 1
            continue
        f = by_idx.loc[i]
        i += 1
        if f.t < a.start:
            continue
        if f.t > t_end:
            break
        lm = lms.get(i - 1)
        if f.face_detected == 1 and lm is not None and np.isfinite(lm).all():
            for x, y in lm.astype(int):
                cv2.circle(frame, (x, y), 1, (255, 255, 0), -1)
            *_, rvec, tvec, K = head_pose(lm, w, h)
            if rvec is not None:
                cv2.drawFrameAxes(frame, K, NO_DISTORTION, rvec, tvec, 150)
            emo = max(EMOTIONS, key=lambda e: f[f"emo_{e}"])
            lines = [f"{emo} {f[f'emo_{emo}']:.2f}   det {f.det_score:.2f}",
                     f"head (ours)  pitch {f.head_pitch:+.0f} yaw {f.head_yaw:+.0f} roll {f.head_roll:+.0f}  err {f.pnp_err:.2f}"]
            if "meta_head_pitch" in f and pd.notna(f.meta_head_pitch):
                lines.append(f"head (data)  pitch {f.meta_head_pitch:+.0f} yaw {f.meta_head_yaw:+.0f} "
                             f"roll {f.meta_head_roll:+.0f}")
            lines += [f"gaze yaw {np.degrees(f.gaze_yaw):+.0f} pitch {np.degrees(f.gaze_pitch):+.0f} deg",
                      f"eye {f.eye_open:.2f} mouth {f.mouth_open:.2f} brow {f.brow_raise:.2f} gap {f.brow_gap:.2f}"]
            text(frame, lines)
            au_bars(frame, f, y0=18 + 17 * len(lines) + 6)
        else:
            text(frame, ["no face"], color=(0, 0, 255))
        hh, mm, ss = int(f.clock // 3600), int(f.clock % 3600 // 60), f.clock % 60
        label = "" if "bored_self" not in f or pd.isna(f.bored_self) else ("BORED (self)" if f.bored_self == 1 else "not bored (self)")
        text(frame, [f"t {f.t:6.1f} s  {hh:02d}:{mm:02d}:{ss:04.1f}", label], x=w - 200, y=18,
             color=(80, 80, 255) if label.startswith("BORED") else (255, 255, 255))

        s = strip.copy()
        x = X(np.array([f.t]))[0]
        cv2.line(s, (x, 0), (x, STRIP_H - 18), (255, 255, 255), 1)
        canvas = np.vstack([frame, s])
        nxt = df.t.values[np.searchsorted(df.t.values, f.t, side="right")] if f.t < df.t.iloc[-1] else f.t + 0.1
        for _ in range(max(1, round((nxt - f.t) * OUT_FPS))):      # repeat to follow the true clock
            wr.write(canvas)
            written += 1
    wr.release()
    print(f"wrote {out} ({written / OUT_FPS:.0f} s of video)")


if __name__ == "__main__":
    main()
