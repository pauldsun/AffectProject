"""DIPSER label analysis, before downloading any video: is there a usable (and forecastable) boredom target, and in
which experiments?

    python src/dipser_labels.py <dipser>/labels --out <dipser>/analysis [--k 2] [--horizon 30]

Sections: 1 data, 2 rater activity, 3 label distributions, 4 agreement, 5 boredom per experiment (+ the size of the
"bored within H s" forecasting task), 6 temporal signal. Writes timeline.csv.gz, recordings.csv, experiments.csv.
Seconds are not independent samples: kappas and rates here are descriptive, not tests.
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import dipser

pd.set_option("display.width", 200, "display.max_columns", 30)


def fleiss(M):
    """Fleiss' kappa; M = items x raters of category values, no NaN."""
    n = M.shape[1]
    counts = np.stack([(M == c).sum(1) for c in np.unique(M)], 1)
    p_i = ((counts ** 2).sum(1) - n) / (n * (n - 1))
    p_j = counts.sum(0) / counts.sum()
    pe = (p_j ** 2).sum()
    return (p_i.mean() - pe) / (1 - pe) if pe < 1 else np.nan


def fixed_n(V, n):
    """Rows of V (items x raters, NaN = rater absent) that have exactly n values -> items x n array.
    Fleiss' kappa needs the same NUMBER of raters per item, not the same raters."""
    keep = (~np.isnan(V)).sum(1) == n
    W = V[keep]
    return W[~np.isnan(W)].reshape(-1, n)


def pairwise(tl, raters, fn):
    """fn(a, b) over the seconds where both raters have a value; NaN if they never overlap."""
    out = pd.DataFrame(np.nan, raters, raters)
    for r1 in raters:
        for r2 in raters:
            m = tl[r1].notna() & tl[r2].notna()
            if m.sum() > 100:
                out.loc[r1, r2] = fn(tl.loc[m, r1].values, tl.loc[m, r2].values)
    return out


def cohen(a, b):
    pe = sum((a == c).mean() * (b == c).mean() for c in np.union1d(a, b))
    return ((a == b).mean() - pe) / (1 - pe) if pe < 1 else np.nan


def onsets(b):
    return np.flatnonzero(b[1:] & ~b[:-1]) + 1


def clean_onsets(b, before=10, after=5):
    """Onsets with >= `before` s not bored before and >= `after` s bored after (drops 1-2 s flickers)."""
    return [i for i in onsets(b) if i >= before and i + after <= len(b) and not b[i - before:i].any()
            and b[i:i + after].all()]


def forecast_targets(b, H):
    """For every second t not bored with H s of future: y = bored at some point in (t, t+H]."""
    c = np.concatenate([[0], np.cumsum(b)])
    t = np.arange(len(b) - H)
    t = t[~b[t]]
    return (c[t + H + 1] - c[t + 1]) > 0


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="folder saved by `dipser_fetch.py labels` (…/dipser/labels)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=2, help="bored = at least k experts say emotion 1")
    ap.add_argument("--horizon", type=int, default=30, help="forecast horizon in seconds")
    ap.add_argument("--drop", default="", help="experts to leave out of the consensus, e.g. e1,e2")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    K, H = a.k, a.horizon

    ev = dipser.load_events(a.root)
    tl = dipser.timelines(ev)
    watch = dipser.watch_files(a.root)
    raters = sorted(ev.rater.unique(), key=lambda r: (r == "self", r))
    dropped = [x for x in a.drop.split(",") if x]
    experts = [r for r in raters if r != "self" and r not in dropped]
    E = [f"emo_{r}" for r in experts]; A = [f"att_{r}" for r in experts]
    keys = ["group", "experiment", "subject"]
    recs = list(tl.groupby(keys, sort=True))

    # ---------------------------------------------------------------- 1
    print("== 1. data")
    n_rec = ev.groupby(keys).ngroups
    print(f"{len(ev)} label entries, {n_rec} recordings, raters {raters}")
    print("recordings per group:", ev.drop_duplicates(keys).groupby("group").size().to_dict())
    have = ev.groupby(keys).rater.agg(set)
    sets = have.map(lambda s: ",".join(r for r in raters if r in s)).rename("raters")
    print("which raters label which recordings (rows = rater set, columns = group):")
    print(sets.reset_index().groupby(["raters", "group"]).size().unstack(fill_value=0).to_string())
    dur = tl.groupby(keys).size() / 60
    print(f"recording length (first label -> last label in the session): min {dur.min():.1f}, "
          f"median {dur.median():.1f}, max {dur.max():.1f} min")
    print("median length per experiment (min):", dur.groupby(level=1).median().round(1).to_dict())
    starts = ev.groupby(keys + ["rater"]).t.min().groupby(keys).agg(lambda x: x.max() - x.min())
    print(f"raters' first entries differ by > 1 s in {(starts > 1).sum()} recordings")
    nw = pd.Series({k: len(watch.get(k, [])) for k in have.index})
    print(f"watch data: {(nw == 0).sum()} recordings without; median {nw[nw > 0].median():.0f} s of watch files")
    ends = dipser.session_ends(ev)
    gap = [watch[k][-1] + 1 - ends[k[:2]] for k in watch if watch[k] and k[:2] in ends]
    if gap:
        print(f"watch end minus label-based session end: median {np.median(gap):+.0f} s "
              f"(IQR {np.percentile(gap, 25):+.0f} .. {np.percentile(gap, 75):+.0f})")

    # ---------------------------------------------------------------- 2
    print("\n== 2. rater activity (entries after the first one = changes)")
    per = ev.groupby(keys + ["rater"]).size().rename("entries").reset_index()
    per["minutes"] = [dur[tuple(k)] for k in per[keys].values]
    per["changes_per_min"] = (per.entries - 1) / per.minutes
    print(per.groupby("rater").agg(median_entries=("entries", "median"),
                                   changes_per_min=("changes_per_min", "median"),
                                   recs_under_3_changes=("entries", lambda x: pct((x - 1 < 3).mean()))))
    first = ev.sort_values(["t", "order"]).groupby(keys + ["rater"]).first()
    print("value each rater gives at the START of a recording (share of recordings; a constant = tool default?):")
    for f in ("emotion", "attention"):
        tab = first.groupby("rater")[f].value_counts(normalize=True).unstack(fill_value=0)
        print(f"{f}:"); print((100 * tab).round(0).to_string())
    for f in ("emotion", "attention"):
        x = ev.dropna(subset=[f]).groupby(keys + ["rater"])[f].nunique()
        print(f"recordings where the rater used only ONE {f} value:",
              (x == 1).groupby(level="rater").mean().map(pct).to_dict())

    # ---------------------------------------------------------------- 3
    print("\n== 3. share of time per value (rows = value, columns = rater)")
    for f, lo, hi in (("emo", 1, 9), ("att", 1, 5)):
        cols = [f"{f}_{r}" for r in raters]
        tab = pd.DataFrame({c[4:]: tl[c].value_counts(normalize=True) for c in cols}).reindex(range(lo, hi + 1))
        print("emotion (" + ", ".join(f"{k} {v}" for k, v in dipser.EMOTIONS.items()) + ")" if f == "emo"
              else "attention (5 = most attentive)")
        print((100 * tab.fillna(0)).round(1).to_string())

    # ---------------------------------------------------------------- 4
    print("\n== 4. agreement (per second, pooled over recordings)")
    nE = int(pd.Series(tl[E].notna().sum(1)).mode()[0])   # experts per recording (4)
    print(f"experts in the consensus: {experts}{' (dropped ' + ','.join(dropped) + ')' if dropped else ''}; "
          f"{nE} of them per recording")
    VE, VA = tl[E].values, tl[A].values
    print(f"Fleiss kappa over the {nE} experts of each second: emotion {fleiss(fixed_n(VE, nE)):.3f}, attention "
          f"{fleiss(fixed_n(VA, nE)):.3f}, bored yes/no {fleiss((fixed_n(VE, nE) == 1).astype(int)):.3f}")
    print("(kappa < 0.2 slight, 0.2-0.4 fair, 0.4-0.6 moderate, > 0.6 substantial)")
    emo = tl[[f"emo_{r}" for r in raters]].set_axis(raters, axis=1)
    att = tl[[f"att_{r}" for r in raters]].set_axis(raters, axis=1)
    bor = (emo == 1).astype(float).where(emo.notna())
    for name, df, fn in (("emotion exact", emo, cohen), ("bored yes/no", bor, cohen),
                         ("attention exact", att, cohen),
                         ("attention within 1 point (% of seconds)", att, lambda x, y: 100 * np.mean(abs(x - y) <= 1))):
        print(f"pairwise {name if '%' in name else 'Cohen kappa, ' + name}:")
        print(pairwise(df, raters, fn).round(2).to_string())
    nb = (VE == 1).sum(1)
    if "emo_self" in tl:
        m = tl["emo_self"].notna().values & (tl[E].notna().sum(1).values == nE)
        s, maj = tl["emo_self"].values[m] == 1, nb[m] >= K
        print(f"self says bored {pct(s.mean())} of the time, >= {K} experts {pct(maj.mean())}; "
              f"kappa self vs expert consensus {cohen(s.astype(int), maj.astype(int)):.3f}")

    # ---------------------------------------------------------------- 5
    print(f"\n== 5. boredom (bored = at least {K} of the {nE} experts present say emotion 1)")
    nb_all = (tl[E].values == 1).sum(1)
    print("share of time with >= k experts saying bored:",
          {k: pct((nb_all >= k).mean()) for k in range(1, nE + 1)})
    rows = []
    for (g, e, s), r in recs:
        b = (r[E].values == 1).sum(1) >= K
        y = forecast_targets(b, H)
        att = r[A].mean(1).values
        rows.append(dict(group=g, experiment=e, subject=s, minutes=len(r) / 60, bored=b.mean(),
                         bored_self=(r["emo_self"].values == 1).mean() if "emo_self" in r else np.nan,
                         onsets=len(onsets(b)), clean_onsets=len(clean_onsets(b)),
                         fc_n=len(y), fc_pos=int(y.sum()), att=np.nanmean(att), low_att=np.mean(att <= 2)))
    rec = pd.DataFrame(rows)
    rec.to_csv(out / "recordings.csv", index=False)

    def table(by):
        t = rec.groupby(by).agg(recordings=("subject", "size"), median_min=("minutes", "median"),
                                bored=("bored", "mean"), bored_self=("bored_self", "mean"),
                                onsets=("onsets", "sum"), clean_onsets=("clean_onsets", "sum"),
                                recs_with_onset=("clean_onsets", lambda x: int((x > 0).sum())),
                                fc_pos=("fc_pos", "sum"), fc_n=("fc_n", "sum"),
                                mean_att=("att", "mean"), low_att=("low_att", "mean"))
        t["fc_rate"] = t.fc_pos / t.fc_n
        for c in ("bored", "bored_self", "low_att", "fc_rate"):
            t[c] = (100 * t[c]).round(1)
        return t.round(2)
    print("columns: bored / bored_self / low_att (expert mean attention <= 2) = % of time; onsets = not bored ->"
          " bored; clean = >= 10 s not bored before and >= 5 s bored after;")
    print(f"fc_pos / fc_n = seconds where the student is not bored and IS bored within {H} s / all not-bored "
          f"seconds with {H} s of future (the forecasting task), fc_rate = its positive rate (%)")
    by_exp = table("experiment"); print(by_exp.to_string())
    print("\nby group and experiment:"); ge = table(["group", "experiment"]); print(ge.to_string())
    ge.to_csv(out / "experiments.csv")
    print(f"\nrecordings with >= 1 clean onset: {(rec.clean_onsets > 0).sum()} of {len(rec)}; "
          f"students (group, subject) with >= 1: {rec[rec.clean_onsets > 0].groupby(['group', 'subject']).ngroups}"
          f" of {rec.groupby(['group', 'subject']).ngroups}")

    # ---------------------------------------------------------------- 6
    print("\n== 6. temporal signal: mean |change| of the expert consensus")
    print("lag L = same recording, L s later. Compare with a random time in the SAME recording, the same student "
          "in another experiment, and another student in the same session (same activity).")
    print("If lag-L ~ 'same recording', there is nothing to forecast beyond the recording's average (DAiSEE).")
    rng = np.random.default_rng(0)
    for name, fn in (("bored share (experts saying 1 / experts present)",
                      lambda r: (r[E].values == 1).sum(1) / np.maximum(r[E].notna().sum(1).values, 1)),
                     ("attention (expert mean)", lambda r: r[A].mean(1).values)):
        x = {k: fn(r) for k, r in recs}
        res = {}
        for L in (1, 5, 10, 30, 60):
            res[f"lag {L}s"] = np.nanmean(np.concatenate([np.abs(v[L:] - v[:-L]) for v in x.values() if len(v) > L]))
        same, stud, sess = [], [], []
        by_student, by_session = {}, {}
        for k in x:
            by_student.setdefault((k[0], k[2]), []).append(k); by_session.setdefault(k[:2], []).append(k)
        for k, v in x.items():
            t = rng.integers(0, len(v), 200)
            same.append(np.abs(v[t] - v[rng.integers(0, len(v), 200)]))
            for pool, acc in ((by_student[(k[0], k[2])], stud), (by_session[k[:2]], sess)):
                others = [o for o in pool if o != k]
                if others:
                    o = x[others[rng.integers(len(others))]]
                    acc.append(np.abs(v[t] - o[rng.integers(0, len(o), 200)]))
        res["random, same recording"] = np.nanmean(np.concatenate(same))
        res["same student, other experiment"] = np.nanmean(np.concatenate(stud)) if stud else np.nan
        res["other student, same session"] = np.nanmean(np.concatenate(sess)) if sess else np.nan
        print(f"{name}:"); print("  " + "\n  ".join(f"{k:32s} {v:.3f}" for k, v in res.items()))

    tl.to_csv(out / "timeline.csv.gz", index=False)
    print(f"\nwrote {out}/timeline.csv.gz ({len(tl)} rows), recordings.csv, experiments.csv")


if __name__ == "__main__":
    main()
