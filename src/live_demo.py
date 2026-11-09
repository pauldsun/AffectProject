"""Live webcam overlay of the shared per-frame features (src/features.py). Run from project root:
   python src/live_demo.py --device mps                 (press q to quit)
   python src/live_demo.py --device mps --record results/live/session1.csv
--record logs the same columns as extract_features.py, timestamped from session start."""
import argparse, time
from pathlib import Path
import cv2, numpy as np, pandas as pd
from features import FeatureExtractor, FEATURE_COLUMNS, EMOTIONS, AU_LABELS, NO_DISTORTION

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
ap.add_argument("--camera", type=int, default=0, help="0 = built-in FaceTime camera")
ap.add_argument("--record", help="CSV path to log per-frame features")
args = ap.parse_args()

print("loading models ...")
ex = FeatureExtractor(device=args.device)

cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise SystemExit(f"could not open camera {args.camera} (check System Settings > Privacy > Camera)")
print("camera open, press q in the video window to quit")


def put(img, lines, color=(255, 255, 255)):
    for k, s in enumerate(lines):
        org = (10, 22 + 20 * k)                 # thick black stroke under the text = readable on light and dark
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_aus(img, f, first_row=5, bar_w=100):
    """All 8 AU scores as bars under the text block. White tick = 0.5 (the 'present' threshold);
    bars past it turn orange. Shows small rises below 0.5 that a top-3 list would hide."""
    for i in range(8):
        y = 22 + 20 * (first_row + i)
        s = float(np.clip(f[f"au_{i}"], 0, 1))
        cv2.rectangle(img, (10, y - 11), (10 + bar_w, y + 3), (0, 0, 0), -1)
        cv2.rectangle(img, (10, y - 11), (10 + int(bar_w * s), y + 3), (0, 165, 255) if s > 0.5 else (200, 200, 200), -1)
        cv2.line(img, (10 + bar_w // 2, y - 13), (10 + bar_w // 2, y + 5), (255, 255, 255), 1)
        label = f"au_{i} {s:.2f}  {AU_LABELS[i]}?"
        for thick, col in ((4, (0, 0, 0)), (1, (255, 255, 255))):
            cv2.putText(img, label, (18 + bar_w, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, thick, cv2.LINE_AA)


rows, misses, t_start = [], 0, time.monotonic()
while True:
    ok, frame = cap.read()
    if not ok:                                  # macOS camera often drops the first frames while warming up
        misses += 1
        if misses > 100:
            raise SystemExit("camera opened but returned no frames")
        cv2.waitKey(30)
        continue
    misses = 0
    t0 = time.monotonic()
    r = ex.process(frame)
    latency = (time.monotonic() - t0) * 1000
    f = r.features
    if args.record:
        rows.append({"clip_id": "live", "frame_idx": len(rows), "t_sec": round(t0 - t_start, 4), **f})

    if r.box is None:
        put(frame, [f"no face  ({latency:.0f} ms)"], (0, 0, 255))
    else:
        x1, y1, x2, y2 = r.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        for x, y in r.landmarks.astype(int):
            cv2.circle(frame, (x, y), 1, (255, 255, 0), -1)
        if r.rvec is not None:                  # red/green/blue = face x/y/z axes from PnP
            cv2.drawFrameAxes(frame, r.K, NO_DISTORTION, r.rvec, r.tvec, 200)
        emo = max(EMOTIONS, key=lambda e: f[f"emo_{e}"])
        put(frame, [
            f"{emo} {f[f'emo_{emo}']:.2f}   latency {latency:.0f} ms",
            f"head pitch {f['head_pitch']:+.0f}  yaw {f['head_yaw']:+.0f}  roll {f['head_roll']:+.0f} deg  (fit err {f['pnp_err']:.2f})",
            f"gaze yaw {np.degrees(f['gaze_yaw']):+.0f}  pitch {np.degrees(f['gaze_pitch']):+.0f} deg",
            f"eye_open {f['eye_open']:.2f}  mouth_open {f['mouth_open']:.2f}  brow_raise {f['brow_raise']:.2f}  brow_gap {f['brow_gap']:.2f}",
        ])
        draw_aus(frame, f)
    cv2.imshow("affect features  (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

if args.record and rows:
    out = Path(args.record)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["clip_id", "frame_idx", "t_sec"] + FEATURE_COLUMNS).to_csv(out, index=False)
    print(f"saved {len(rows)} frames to {out}")
