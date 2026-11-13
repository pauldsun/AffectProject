"""DIPSER/DIPSEER labels: change-point JSONs -> per-second timelines.

Layout (as saved by dipser_fetch.py labels): .../group_GG/experiment_EE/subject_SS/labels/<rater>.json, raters
labeler_01..05 (experts, 4 per recording) + self_labeling. Each file is a list of {"datetime": "HH:MM:SS:ffffff", "attention"?,
"emotion"?}; a field missing from an entry means it did not change. Values hold until the next entry that has them.
Codes (paper, Pekrun et al.): emotion 9 enjoyment, 8 hope, 7 pride, 6 relief, 5 anger, 4 anxiety, 3 shame,
2 despair, 1 boredom; attention 1..5, 5 = maximum. Up to 5 expert files (labeler_01..05) exist, 4 per recording.

Our own loader instead of the official one: that one forward-fills across subjects and drops same-time tags.
"""
import json, re
from pathlib import Path
import numpy as np, pandas as pd

FIELDS = ("attention", "emotion")
EMOTIONS = {9: "enjoyment", 8: "hope", 7: "pride", 6: "relief", 5: "anger", 4: "anxiety", 3: "shame",
            2: "despair", 1: "boredom"}
BOREDOM = 1
REC_RE = re.compile(r"group_(\d+)/experiment_(\d+)/subject_(\d+)")


def clock(s):
    """'HH:MM:SS:ffffff' (or ':mmm', or '_' separated) -> seconds since midnight."""
    h, m, sec, frac = re.split(r"[:_]", s)
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(frac) / 10 ** len(frac)


def rater_name(stem):
    if stem.startswith("self"):
        return "self"
    m = re.search(r"(\d+)$", stem)
    return f"e{int(m.group(1))}" if m else stem


def _num(v):
    return float(v) if v not in (None, "") else np.nan


def label_rows(entries, rater, **ids):
    """One label file's entries (parsed JSON list) -> event dicts."""
    return [dict(**ids, rater=rater, order=i, t=clock(x["datetime"]), attention=_num(x.get("attention")),
                 emotion=_num(x.get("emotion"))) for i, x in enumerate(entries)]


def load_events(root):
    """All label entries: group, experiment, subject, rater, order, t (s since midnight), attention, emotion."""
    rows = []
    for f in sorted(Path(root).glob("**/labels/*.json")):
        m = REC_RE.search(f.as_posix())
        if not m:
            continue
        g, e, s = map(int, m.groups())
        rows += label_rows(json.load(open(f)), rater_name(f.stem), group=g, experiment=e, subject=s)
    return pd.DataFrame(rows)


def values_at(rec, times):
    """Label values in force at each time (s since midnight) for one recording's events:
    {"att_<rater>": array, "emo_<rater>": array}, NaN before a rater's first value for that field."""
    times = np.asarray(times, dtype=float)
    out = {}
    for r, rr in rec.groupby("rater"):
        for f in FIELDS:
            x = rr.dropna(subset=[f]).sort_values(["t", "order"], kind="stable")
            v = np.full(len(times), np.nan)
            if len(x):
                i = np.searchsorted(x.t.values, times, side="right") - 1   # last entry at or before t
                v[i >= 0] = x[f].values[i[i >= 0]]
            out[f"{f[:3]}_{r}"] = v
    return out


def watch_files(root):
    """{(group, experiment, subject): sorted start times of the 1 s watch files}."""
    out = {}
    for d in Path(root).glob("**/watch_sensor*"):
        m = REC_RE.search(d.as_posix())
        if d.is_dir() and m:
            out[tuple(map(int, m.groups()))] = sorted(clock(f.stem) for f in d.glob("*.json"))
    return out


def session_ends(ev):
    """End of each class session (group, experiment) = last label entry by anyone in it. Labels say nothing
    about when recording stopped; with ~20 students x 5 raters, someone changes a label near the end."""
    return ev.groupby(["group", "experiment"]).t.max().to_dict()


def timelines(ev, ends=None, step=1.0):
    """Wide per-step table: group, experiment, subject, t (s from the recording's first label), clock, and
    att_<rater> / emo_<rater> holding the value in force at that time (NaN before a rater's first value)."""
    ends = ends if ends is not None else session_ends(ev)
    out = []
    for (g, e, s), rec in ev.groupby(["group", "experiment", "subject"], sort=True):
        t0 = rec.t.min()
        t1 = max(ends.get((g, e), rec.t.max()), rec.t.max())
        grid = t0 + np.arange(0, t1 - t0 + 1e-9, step)
        df = pd.DataFrame({"group": g, "experiment": e, "subject": s, "t": grid - t0, "clock": grid})
        for c, v in values_at(rec, grid).items():
            df[c] = v
        out.append(df)
    return pd.concat(out, ignore_index=True)
