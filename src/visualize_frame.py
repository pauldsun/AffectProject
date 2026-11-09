"""Draw OpenFace CSV results over the source image.
Usage (from project root):  python src/visualize_frame.py <path/to/face_analysis_*.csv>
"""
import ast, sys
from math import sin, radians
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt

EMOTIONS = {0: "Neutral", 1: "Happy", 2: "Sad", 3: "Surprise",
            4: "Fear", 5: "Disgust", 6: "Anger", 7: "Contempt"}
# WFLW 98-point regions (indices from OpenFace's STAR alignment config)
REGIONS = {"contour": (range(0, 33), "gray"), "R brow": (range(33, 42), "brown"),
           "L brow": (range(42, 51), "brown"), "nose": (range(51, 60), "orange"),
           "R eye": (range(60, 68), "cyan"), "L eye": (range(68, 76), "cyan"),
           "outer lip": (range(76, 88), "red"), "inner lip": (range(88, 96), "red"),
           "pupils": (range(96, 98), "lime")}

csv_path = Path(sys.argv[1])
row = pd.read_csv(csv_path).iloc[0]          # first frame/face; loop over rows for video

# image_path in the CSV is relative to wherever `detect` was run — resolve it
img_path = Path(row["image_path"])
if not img_path.exists():
    img_path = csv_path.parent.parent / row["image_path"]   # e.g. results/../images/0.jpg

lm = ast.literal_eval(row["landmarks"])      # CSV stores lists as strings → parse back
img = plt.imread(img_path)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img)
for name, (idxs, color) in REGIONS.items():
    xs = [lm[i][0] for i in idxs]; ys = [lm[i][1] for i in idxs]
    ax.scatter(xs, ys, s=6, c=color, label=name)

# gaze rays from the pupils (sign conventions UNVERIFIED — see note below)
L = 60
for p in (96, 97):
    ax.annotate("", xy=(lm[p][0] + L * sin(radians(row["gaze_yaw"])),
                        lm[p][1] - L * sin(radians(row["gaze_pitch"]))),
                xytext=lm[p], arrowprops=dict(arrowstyle="->", color="magenta", lw=2))

aus = ast.literal_eval(row["action_units"])
aus = aus[0] if aus and isinstance(aus[0], list) else aus   # unwrap batch dim: [[v...]] → [v...]
ax.set_title(f"{EMOTIONS[int(row['emotion'])]} | yaw {row['gaze_yaw']:.1f}° pitch {row['gaze_pitch']:.1f}°\n"
             f"AUs: " + ", ".join(f"#{i}:{v:.2f}" for i, v in enumerate(aus)))
ax.legend(loc="lower right", fontsize=7)
out = csv_path.with_suffix(".png")
plt.savefig(out, bbox_inches="tight"); print(f"saved {out}")