"""DAiSEE layout and label helpers shared by audit_continuity.py and sample_clips.py.
Expected layout (the official zip):  <root>/{Train,Validation,Test}/<subject>/<clip>/<clip>.avi
Also accepts <root>/<subject>/<clip>/<clip>.avi and <root>/<subject>/<clip>.avi."""
from pathlib import Path
import pandas as pd

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}
AFFECTS = ["Boredom", "Engagement", "Confusion", "Frustration"]


def find_clips(root):
    """DataFrame [split, subject, clip_id, path], sorted by subject then clip number (numeric when possible),
    so consecutive rows of one subject are the candidate 'clip N, clip N+1' pairs."""
    root = Path(root)
    rows = []
    for v in (p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS):
        parts = v.relative_to(root).parts
        nested = len(parts) >= 3 and parts[-2] == v.stem          # .../<subject>/<clip>/<clip>.avi
        subject = parts[-3] if nested else (parts[-2] if len(parts) >= 2 else "unknown")
        split = parts[0] if parts[0] != subject and len(parts) >= (4 if nested else 3) else ""
        rows.append(dict(split=split, subject=subject, clip_id=v.stem, path=str(v)))
    df = pd.DataFrame(rows, columns=["split", "subject", "clip_id", "path"])
    df["_num"] = pd.to_numeric(df.clip_id, errors="coerce")
    return df.sort_values(["subject", "_num", "clip_id"]).drop(columns="_num").reset_index(drop=True)


def load_labels(path):
    """AllLabels.csv -> DataFrame [clip_id, Boredom, Engagement, Confusion, Frustration].
    The official file names clips 'xxxx.avi' and has a trailing space in 'Frustration '."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["clip_id"] = df["ClipID"].astype(str).str.strip().map(lambda s: Path(s).stem)
    return df[["clip_id"] + AFFECTS]
