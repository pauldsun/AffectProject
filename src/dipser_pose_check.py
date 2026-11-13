"""DIPSER: validate our PnP head pose against the dataset's own per-frame face analysis.

Three head-pose estimates per frame:
  ours      our PnP on our STAR landmarks (features/<rec>/images.csv, head_pitch/yaw/roll)
  mesh      the SAME PnP (same 3D face model, same camera guess) run on the dataset's MediaPipe face mesh points
            (metadata facemesh, 478 points normalised to the dataset's face box; mapped back to pixels here)
  provided  the dataset's ready-made head-pose angles (metadata headpose.pose)

ours vs mesh tests our landmarks: two independent detectors fed through one PnP should give the same angles.
It does not test the 3D model or the camera guess, which both share; those mostly set the absolute scale and
zero point, not the motion over time. provided vs mesh tests the dataset's own angles against its own mesh.
Also reported: how far our 6 PnP landmarks are from the matching mesh points, in inter-ocular distances (IOD).

Writes <out>/pose_check.csv (one row per recording) and, with --plot REC, a timeline figure for one recording.

  python src/dipser_pose_check.py --data data/dipser --out results/pose_check --include experiment_03
  python src/dipser_pose_check.py --data data/dipser --out results/pose_check \
      --include experiment_03/subject_15 --plot DIPSEER/group_03/experiment_03/subject_15
"""
import argparse, json, sys, tarfile
from pathlib import Path
import cv2, numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import PNP_IDX, head_pose

# Our WFLW-98 PnP points -> the matching MediaPipe face-mesh indices (checked on DIPSER frames: 0.01-0.1 IOD apart)
#             nose tip  chin  eye outer (image left)  eye outer (image right)  mouth corner L  mouth corner R
MESH_IDX = dict(zip(PNP_IDX, [1, 152, 33, 263, 61, 291]))
ANGLES = ("pitch", "yaw", "roll")


def mesh_points(face):
    """Dataset face metadata -> {our landmark index: (x, y) in pixels}, or None."""
    fm, box = face.get("facemesh"), face.get("bounding_box")
    if isinstance(box, (list, tuple)) and len(box) == 4:
        box = dict(zip(("x0", "y0", "x1", "y1"), box))
    if not fm or len(fm) < 300 or not isinstance(box, dict):
        return None
    w, h = box["x1"] - box["x0"], box["y1"] - box["y0"]
    return {o: (box["x0"] + fm[m]["x"] * w, box["y0"] + fm[m]["y"] * h) for o, m in MESH_IDX.items()}


def check(rec, root):
    vid, feat = root / "videos" / rec, root / "features" / rec
    cap = cv2.VideoCapture(str(vid / "images.mp4"))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    fr = pd.read_csv(vid / "images.frames.csv")
    frame_of = dict(zip(fr.file.map(lambda f: Path(f).stem), fr.video_frame))
    ours = pd.read_csv(feat / "images.csv").set_index("frame_idx")
    z = np.load(feat / "images.npz")
    lms = dict(zip(z["frame_idx"].tolist(), z["landmarks"]))

    rows = []
    with tarfile.open(vid / "extras.tar.gz", "r:gz") as tf:
        for m in tf:
            if not (m.isfile() and "metadata" in Path(m.name).parts and m.name.endswith(".json")):
                continue
            k = frame_of.get(Path(m.name).stem)
            face = ((json.load(tf.extractfile(m)) or {}).get("person") or {}).get("face") or {}
            pts = mesh_points(face)
            if k is None or pts is None or k not in ours.index or ours.at[k, "face_detected"] != 1:
                continue
            L = lms.get(k)
            if L is None or not np.isfinite(L).all():
                continue
            pseudo = np.full((98, 2), np.nan, np.float64)          # only the PnP points are needed
            for o, xy in pts.items():
                pseudo[o] = xy
            mp, my, mr, *_ = head_pose(pseudo, W, H)
            iod = np.linalg.norm(L[60] - L[72])
            err = np.mean([np.linalg.norm(L[o] - np.array(xy)) for o, xy in pts.items()]) / iod
            pose = (face.get("headpose") or {}).get("pose") or {}
            rows.append(dict(frame_idx=k, lm_err=err, mesh_pitch=mp, mesh_yaw=my, mesh_roll=mr,
                             prov_pitch=pose.get("pitch", np.nan), prov_yaw=pose.get("yaw", np.nan),
                             prov_roll=pose.get("roll", np.nan)))
    d = pd.DataFrame(rows).set_index("frame_idx").join(ours[[f"head_{a}" for a in ANGLES]]).sort_index()
    d = d.rename(columns={f"head_{a}": f"ours_{a}" for a in ANGLES})
    d["t"] = fr.set_index("video_frame").time_s.reindex(d.index).values
    d["t"] -= fr.time_s.iloc[0]
    return d


def summarize(d):
    s = dict(frames=len(d), lm_err_iod=round(d.lm_err.median(), 3))
    for a in ANGLES:
        o, m, p = d[f"ours_{a}"], d[f"mesh_{a}"], d[f"prov_{a}"]
        s[f"r_{a}_ours_mesh"] = round(o.corr(m), 2)
        s[f"mae_{a}_ours_mesh"] = round((o - m).abs().median(), 1)        # degrees
        s[f"r_{a}_prov_mesh"] = round(p.corr(m), 2)
        s[f"r_{a}_ours_prov"] = round(o.corr(p), 2)
        s[f"sd_{a}_ours"] = round(o.std(), 1)
    # the provided angles look axis-swapped: their "pitch" follows -yaw and their "yaw" follows -pitch
    s["r_provpitch_mesh_yaw"] = round(d.prov_pitch.corr(d.mesh_yaw), 2)
    s["r_provyaw_mesh_pitch"] = round(d.prov_yaw.corr(d.mesh_pitch), 2)
    return s


def plot(d, rec, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True)
    for ax, a in zip(axes, ANGLES):
        ax.plot(d.t, d[f"ours_{a}"], lw=0.9, color="#1f5fbf", label="ours (PnP on our landmarks)")
        ax.plot(d.t, d[f"mesh_{a}"], lw=0.9, color="#e08a1e", alpha=0.85, label="PnP on dataset face mesh")
        ax.plot(d.t, d[f"prov_{a}"], lw=0.7, color="#888888", alpha=0.8, label="dataset's provided angles")
        ax.set_ylabel(f"{a} (deg)")
        ax.grid(alpha=0.25)
        ax.set_title(f"{a}: r(ours, mesh) = {d[f'ours_{a}'].corr(d[f'mesh_{a}']):.2f}, "
                     f"r(provided, mesh) = {d[f'prov_{a}'].corr(d[f'mesh_{a}']):.2f}", fontsize=9, loc="left")
    axes[0].legend(fontsize=8, ncol=3, loc="upper right", frameon=False)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Head pose check, {rec}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/dipser", help="folder with videos/ and features/")
    ap.add_argument("--out", default="results/pose_check")
    ap.add_argument("--include", default="", help="only recordings whose path contains this")
    ap.add_argument("--plot", help="recording to draw a timeline for, e.g. DIPSEER/group_03/experiment_03/subject_15")
    a = ap.parse_args()

    root, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = sorted(p.parent.relative_to(root / "features") for p in (root / "features").glob("**/images.csv")
                  if a.include in p.as_posix() and (root / "videos" / p.parent.relative_to(root / "features")
                                                    / "extras.tar.gz").exists())
    print(f"{len(recs)} recordings with features and extras")
    rows = []
    for i, rec in enumerate(recs, 1):
        try:
            d = check(rec, root)
        except Exception as ex:
            print(f"[{i}/{len(recs)}] {rec}: FAILED {type(ex).__name__}: {ex}")
            continue
        s = summarize(d)
        rows.append(dict(recording=rec.as_posix(), **s))
        print(f"[{i}/{len(recs)}] {rec}: {s['frames']} frames, landmarks {s['lm_err_iod']} IOD apart, r(ours, mesh) "
              f"pitch {s['r_pitch_ours_mesh']} yaw {s['r_yaw_ours_mesh']} roll {s['r_roll_ours_mesh']}")
        if a.plot and rec.as_posix() == Path(a.plot).as_posix():
            f = out / f"{rec.as_posix().replace('/', '_')}.png"
            plot(d, rec.as_posix(), f)
            print(f"  plot -> {f}")
    if not rows:
        return
    s = pd.DataFrame(rows)
    s.to_csv(out / "pose_check.csv", index=False)
    q = lambda c: f"{s[c].median():.2f} [{s[c].quantile(0.1):.2f}, {s[c].min():.2f}]"
    print(f"\n== {len(s)} recordings, {s.frames.sum()} frames -> {out}/pose_check.csv")
    print(f"our 6 PnP landmarks vs the mesh: median {s.lm_err_iod.median():.3f} IOD apart "
          f"(worst recording {s.lm_err_iod.max():.3f})")
    print("correlation per recording, median [10th percentile, min]:")
    print(f"  {'':6s} {'ours vs mesh':22s} {'provided vs mesh':22s} {'ours vs provided':22s} ours-mesh median |diff|")
    for k in ANGLES:
        print(f"  {k:6s} {q(f'r_{k}_ours_mesh'):22s} {q(f'r_{k}_prov_mesh'):22s} {q(f'r_{k}_ours_prov'):22s} "
              f"{s[f'mae_{k}_ours_mesh'].median():.1f} deg")
    print("axis-swap test (the dataset's provided angles against the other mesh axis):")
    print(f"  provided 'pitch' vs mesh yaw:   {q('r_provpitch_mesh_yaw')}")
    print(f"  provided 'yaw'   vs mesh pitch: {q('r_provyaw_mesh_pitch')}")
    print("  (strongly negative = the provided pitch and yaw are swapped, with flipped signs)")


if __name__ == "__main__":
    main()
