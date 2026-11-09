"""Copy a small, viewable subset of raw DAiSEE clips (e.g. from Colab to Drive, then to the Mac), keeping the
dataset's folder layout so extract_features.py / audit_continuity.py run on it unchanged. Run from project root:
   python src/sample_clips.py data/DAiSEE/DataSet --labels data/DAiSEE/Labels/AllLabels.csv \\
       --out /content/drive/MyDrive/AffectProject/sample_clips

Picks (fixed --seed, so it is reproducible):
  --full N        subjects copied completely (every clip, in order: whole-session continuity, all label changes)
  --subjects M    other subjects, one run of --run consecutive clips each; with --labels the run with the
                  most label changes is chosen (the interesting moments to look at), else a random run
Writes manifest.csv (what was copied and why) and copies the labels file alongside."""
import argparse, random, shutil
from pathlib import Path
import pandas as pd
from daisee import find_clips, load_labels, AFFECTS


def best_run(g, k, labels, rng):
    """Start index of the k-clip run with the most label changes between neighbours (ties -> random)."""
    if len(g) <= k:
        return 0
    starts = list(range(len(g) - k + 1))
    if labels is None:
        return rng.choice(starts)
    lab = g[["clip_id"]].merge(labels, on="clip_id", how="left")[AFFECTS]
    changes = lab.diff().abs().fillna(0).sum(axis=1).to_numpy()   # changes[i] = clip i-1 -> clip i
    score = [changes[s + 1: s + k].sum() for s in starts]
    top = max(score)
    return rng.choice([s for s, v in zip(starts, score) if v == top])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="DAiSEE DataSet folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", help="AllLabels.csv (picks runs with label changes; copied to --out)")
    ap.add_argument("--full", type=int, default=2, help="subjects to copy completely")
    ap.add_argument("--subjects", type=int, default=20, help="other subjects to take one run from")
    ap.add_argument("--run", type=int, default=6, help="consecutive clips per run (6 = 1 minute)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    clips = find_clips(root)
    labels = load_labels(args.labels) if args.labels else None
    rng = random.Random(args.seed)
    subs = list(clips.subject.unique())
    rng.shuffle(subs)

    picks = []
    for i, s in enumerate(subs[: args.full + args.subjects]):
        g = clips[clips.subject == s].reset_index(drop=True)
        if i < args.full:
            picks.append(g.assign(reason="full subject"))
        else:
            st = best_run(g, args.run, labels, rng)
            picks.append(g.iloc[st: st + args.run].assign(reason=f"run of {args.run}"))
    sel = pd.concat(picks, ignore_index=True)

    total = 0
    for p in sel.path:
        src = Path(p)
        dst = out / src.relative_to(root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)                          # copy2 keeps mtime (the audit records it)
        total += src.stat().st_size
    sel.assign(path=[str(Path(p).relative_to(root)) for p in sel.path]).to_csv(out / "manifest.csv", index=False)
    if args.labels:
        shutil.copy2(args.labels, out / Path(args.labels).name)
    print(f"copied {len(sel)} clips from {sel.subject.nunique()} subjects ({total / 1e9:.2f} GB) to {out}")


if __name__ == "__main__":
    main()
