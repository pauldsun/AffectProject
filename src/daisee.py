"""DAiSEE layout and label helpers shared by audit_continuity.py and sample_clips.py.
Expected layout (the official zip):  <root>/{Train,Validation,Test}/<subject>/<clip>/<clip>.avi
Also accepts <root>/<subject>/<clip>/<clip>.avi and <root>/<subject>/<clip>.avi."""
from pathlib import Path
import pandas as pd

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}
AFFECTS = ["Boredom", "Engagement", "Confusion", "Frustration"]


def parse_clip_ids(subject, clip_ids):
    """One subject's clip ids -> {clip_id: (session, clip number)}; ids that don't fit -> ("", None).

    A clip id is the subject number followed by a session digit and a clip number, e.g. 1100011083 =
    subject 110001, session 1, clip 83. Some subjects write it differently, always for all their clips:
    unpadded clip numbers (24851013 = session 1, clip 3), an extra leading 0 (1813740276 = session 2, clip 76),
    a trailing 0 on every id (29042802790 = session 2, clip 79), and 907001 has no session digit
    (90700150 ... 9070011110 = clips 5 ... 111). Verified on the cached audit thumbnails (2026-11-10):
    step-1 neighbours under this reading join seamlessly, a missing clip (step 2) shows a clear jump."""
    suf = {c: c[len(subject):] for c in clip_ids if c.startswith(subject) and c[len(subject):].isdigit()}
    if suf and all(s.startswith("0") for s in suf.values()):
        suf = {c: s[1:] for c, s in suf.items()}
    if suf and all(s.endswith("0") for s in suf.values()):
        suf = {c: s[:-1] for c, s in suf.items()}
    if suf and min(map(len, suf.values())) < 2:          # too short for a session digit (907001)
        parsed = {c: ("1", int(s)) for c, s in suf.items()}
    else:
        parsed = {c: (s[0], int(s[1:])) for c, s in suf.items()}
    return {c: parsed.get(c, ("", None)) for c in clip_ids}


def find_clips(root):
    """DataFrame [split, subject, clip_id, session, clip_num, path], sorted by subject, session, clip number
    (see parse_clip_ids), so consecutive rows of one subject are the candidate 'clip N, clip N+1' pairs."""
    root = Path(root)
    rows = []
    for v in (p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS):
        parts = v.relative_to(root).parts
        nested = len(parts) >= 3 and parts[-2] == v.stem          # .../<subject>/<clip>/<clip>.avi
        subject = parts[-3] if nested else (parts[-2] if len(parts) >= 2 else "unknown")
        split = parts[0] if parts[0] != subject and len(parts) >= (4 if nested else 3) else ""
        rows.append(dict(split=split, subject=subject, clip_id=v.stem, path=str(v)))
    df = pd.DataFrame(rows, columns=["split", "subject", "clip_id", "path"])
    parsed = {}
    for s, g in df.groupby("subject"):
        parsed.update(parse_clip_ids(s, g.clip_id.tolist()))
    df["session"] = df.clip_id.map(lambda c: parsed[c][0])
    df["clip_num"] = df.clip_id.map(lambda c: parsed[c][1]).astype("Int64")
    df = df[["split", "subject", "clip_id", "session", "clip_num", "path"]]
    return df.sort_values(["subject", "session", "clip_num", "clip_id"]).reset_index(drop=True)


def load_labels(path):
    """AllLabels.csv -> DataFrame [clip_id, Boredom, Engagement, Confusion, Frustration].
    The official file names clips 'xxxx.avi' and has a trailing space in 'Frustration '."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["clip_id"] = df["ClipID"].astype(str).str.strip().map(lambda s: Path(s).stem)
    return df[["clip_id"] + AFFECTS]
