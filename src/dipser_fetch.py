"""DIPSER/DIPSEER over FTP without storing the 668 GB: list it, read only the labels, then download -> shrink -> delete.
Run from the project root (on Colab: see dipser_colab.ipynb). Credentials come from environment variables
DIPSER_HOST, DIPSER_PORT, DIPSER_USER, DIPSER_PASS, so they never appear in commands, notebooks or logs.

  python src/dipser_fetch.py list   --out OUT                       # manifest.csv of every remote file + size summary
  python src/dipser_fetch.py labels --out OUT [--include RE]        # label + watch files read out of each remote zip
  python src/dipser_fetch.py fetch  --out OUT --work /content/w --include RE [--exclude context_cam] [--limit N] [--dry-run]

fetch, per zip: download (resumable) to --work -> every folder of frames becomes <folder>.mp4 (H.264, --crf 18,
visually lossless) + <folder>.frames.csv (original frame file names and times: the cameras run at "about 10 fps",
and labels refer to exact frame times) -> everything that is not a frame (labels, metadata, watch sensors) goes
into extras.tar.gz -> done.json -> the download and the frames are deleted. Finished zips are skipped on rerun.
"""
import argparse, csv, ftplib, io, json, os, re, shutil, subprocess, sys, tarfile, time, zipfile
from collections import defaultdict
from pathlib import Path

IMG_EXTS = (".png", ".jpg", ".jpeg")
LABEL_RE = r"(^|/)(labels?|watch_sensors?)/"


# ---------- FTP ----------
def connect(tls=False):
    env = {k: os.environ.get(f"DIPSER_{k}") for k in ("HOST", "PORT", "USER", "PASS")}
    if not all(env.values()):
        sys.exit("set DIPSER_HOST, DIPSER_PORT, DIPSER_USER, DIPSER_PASS first (Colab: from the Secrets panel)")
    conn = (ftplib.FTP_TLS if tls else ftplib.FTP)(timeout=120)
    conn.connect(env["HOST"], int(env["PORT"]))
    conn.login(env["USER"], env["PASS"])
    if tls:
        conn.prot_p()
    conn.voidcmd("TYPE I")
    return conn


def parse_list_line(line):
    """One LIST line -> (name, is_dir, size) for Unix-style or Windows/IIS-style listings; None if unparseable."""
    parts = line.split(None, 8)
    if len(parts) == 9 and parts[0][:1] in "d-l":            # drwxr-xr-x 1 owner group size Mon DD HH:MM name
        if parts[0][0] == "l":
            return None
        return parts[8], parts[0][0] == "d", int(parts[4]) if parts[4].isdigit() else -1
    parts = line.split(None, 3)
    if len(parts) == 4 and re.match(r"\d\d-\d\d-\d\d", parts[0]):   # 01-31-25  10:32AM  <DIR>  name
        return parts[3], parts[2] == "<DIR>", -1 if parts[2] == "<DIR>" else int(parts[2])
    return None


def list_dir(conn, path, method=[None]):
    """[(name, is_dir, size)] for one folder. Tries MLSD, then LIST, then NLST + SIZE; remembers what worked."""
    tries = [method[0]] if method[0] else ["mlsd", "list", "nlst"]
    for m in tries:
        try:
            if m == "mlsd":
                out = [(n, f.get("type") == "dir", int(f.get("size", -1))) for n, f in conn.mlsd(path, facts=["type", "size"])
                       if f.get("type") in ("dir", "file")]
            elif m == "list":
                conn.cwd(path)
                lines = []
                conn.retrlines("LIST", lines.append)
                out = [e for e in map(parse_list_line, lines) if e]
                if lines and not out:
                    print("  unrecognised LIST format, e.g.:", lines[:3])
                    continue
            else:
                conn.cwd(path)
                names = [n.rstrip("/").split("/")[-1] for n in conn.nlst()]
                conn.voidcmd("TYPE I")             # listing switched to text mode; SIZE needs binary
                out = []
                for n in names:
                    try:                           # a folder if we can enter it
                        conn.cwd(f"{path.rstrip('/')}/{n}")
                        out.append((n, True, -1))
                    except ftplib.error_perm:
                        try:
                            out.append((n, False, conn.size(f"{path.rstrip('/')}/{n}")))
                        except ftplib.error_perm:
                            out.append((n, False, -1))
            if method[0] is None:
                print(f"  (listing with {m.upper()})")
            method[0] = m
            return [e for e in out if e[0] not in (".", "..")]
        except ftplib.error_perm as e:
            last = e
    raise last


def walk(conn, path):
    """Yield (path, size) for every file under path. Folders this account may not open are skipped."""
    try:
        entries = list_dir(conn, path)
    except ftplib.error_perm as e:
        print(f"  skipped {path}: {e}")
        return
    if path.count("/") <= 2:
        print(f"  {path}: {', '.join(n + ('/' if d else '') for n, d, _ in entries[:12])}"
              f"{' ...' if len(entries) > 12 else ''}", flush=True)
    for name, is_dir, size in entries:
        p = f"{path.rstrip('/')}/{name}"
        if is_dir:
            yield from walk(conn, p)
        else:
            yield p, size


class FTPRange(io.RawIOBase):
    """Seekable read-only view of a remote file: each read is one RETR starting at an offset (REST), aborted once
    enough bytes arrived. Lets zipfile read the table of contents and single members without the whole archive."""
    def __init__(self, conn_fn, path, size):
        self.conn_fn, self.path, self.size, self.pos, self.conn = conn_fn, path, size, 0, conn_fn()
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos
    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        for attempt in range(3):
            try:
                data = self._fetch(self.pos, n)
                break
            except (OSError, EOFError, ftplib.Error):
                if attempt == 2:
                    raise
                self.conn = self.conn_fn()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)
    def _fetch(self, off, n):
        sock = self.conn.transfercmd(f"RETR {self.path}", rest=off)
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(min(1 << 20, n - len(buf)))
            if not chunk:
                break
            buf += chunk
        sock.close()
        try:                                 # server answers 226 (complete) or 426/451 (we hung up early)
            self.conn.voidresp()
        except ftplib.Error:
            pass
        return bytes(buf)


def download(conn_fn, path, size, dst, tries=5):
    """Resumable binary download with a progress line (MB/s)."""
    from tqdm import tqdm
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(tries):
        have = dst.stat().st_size if dst.exists() else 0
        if have == size:
            return
        try:
            conn = conn_fn()
            with open(dst, "ab") as f, tqdm(total=size, initial=have, unit="B", unit_scale=True, unit_divisor=1024,
                                             desc=Path(path).name, mininterval=5) as bar:
                def write(chunk):
                    f.write(chunk); bar.update(len(chunk))
                conn.retrbinary(f"RETR {path}", write, blocksize=1 << 20, rest=have or None)
            conn.quit()
        except ftplib.error_perm as e:
            if have and "rest" in str(e).lower() or have and str(e).startswith(("500", "502", "504")):
                print(f"  server refused to resume ({e}); restarting {Path(path).name} from the beginning")
                dst.unlink()
                continue
            raise
        except (OSError, EOFError, ftplib.Error) as e:
            print(f"  connection dropped ({e}); resuming ({attempt + 1}/{tries})")
            time.sleep(5)
    if not dst.exists() or dst.stat().st_size != size:
        raise RuntimeError(f"download of {path} incomplete")


# ---------- local processing ----------
def frame_key(name):
    """'10_32_15_250000.png' -> (10, 32, 15, 250000); unparseable names sort by name after parseable ones."""
    parts = Path(name).stem.split("_")
    return (0, tuple(int(p) for p in parts), name) if parts and all(p.isdigit() for p in parts) else (1, (), name)


def seconds(key):
    v = key[1]
    return v[0] * 3600 + v[1] * 60 + v[2] + v[3] / 1e6 if len(v) == 4 else ""


def encode_folder(folder, frames, mp4, crf, fps):
    """Frames (sorted) -> H.264 mp4 at a constant fps; real frame times go to the .frames.csv instead."""
    seq = folder / "__seq"
    seq.mkdir()
    ext = Path(frames[0]).suffix.lower()
    for i, f in enumerate(frames):
        os.symlink(os.path.abspath(folder / f), seq / f"{i:06d}{ext}")
    cmd = ["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(seq / f"%06d{ext}"),
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
           "-pix_fmt", "yuv420p", str(mp4)]
    subprocess.run(cmd, check=True)
    shutil.rmtree(seq)


def process_zip(zpath, out_dir, work, crf, fps):
    root = work / "x"
    shutil.rmtree(root, ignore_errors=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(root)
    stats = dict(videos={}, zip_mb=round(zpath.stat().st_size / 1e6, 1))
    frame_dirs = defaultdict(list)
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                frame_dirs[Path(dirpath)].append(f)
    tmp_out = work / "out"
    shutil.rmtree(tmp_out, ignore_errors=True); tmp_out.mkdir()
    for d, files in sorted(frame_dirs.items()):
        files.sort(key=frame_key)
        name = str(d.relative_to(root)).replace("/", "__") or "frames"
        encode_folder(d, files, tmp_out / f"{name}.mp4", crf, fps)
        keys = [frame_key(f) for f in files]
        with open(tmp_out / f"{name}.frames.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["video_frame", "file", "time_s"])
            w.writerows((i, f, seconds(k)) for i, (f, k) in enumerate(zip(files, keys)))
        t = [seconds(k) for k in keys if k[0] == 0]
        stats["videos"][name] = dict(frames=len(files), mb=round((tmp_out / f"{name}.mp4").stat().st_size / 1e6, 1),
                                     real_fps=round((len(t) - 1) / (t[-1] - t[0]), 2) if len(t) > 1 and t[-1] > t[0] else None)
        for f in files:                      # frames are in the video now
            (d / f).unlink()
    with tarfile.open(tmp_out / "extras.tar.gz", "w:gz") as tar:   # labels, metadata, watch sensors, anything else
        tar.add(root, arcname=".")
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in tmp_out.iterdir():
        shutil.move(str(f), out_dir / f.name)
    shutil.rmtree(root); shutil.rmtree(tmp_out)
    return stats


# ---------- commands ----------
def load_manifest(out):
    m = Path(out) / "manifest.csv"
    if not m.exists():
        sys.exit(f"{m} not found: run the list command first")
    with open(m) as f:
        return [(r["path"], int(r["size"])) for r in csv.DictReader(f)]


def select(rows, include, exclude, ext=None):
    return [(p, s) for p, s in rows if (not include or re.search(include, p)) and
            not (exclude and re.search(exclude, p)) and (ext is None or p.lower().endswith(ext))]


def cmd_list(a):
    conn = connect(a.tls)
    root = a.root or conn.pwd()
    print("server:", conn.getwelcome().splitlines()[0][:120], "| login folder:", conn.pwd(), "| listing from:", root)
    try:
        print("server features:", " ".join(l.strip() for l in conn.sendcmd("FEAT").splitlines()[1:-1]))
    except ftplib.Error:
        pass
    rows = []
    for i, r in enumerate(walk(conn, root), 1):
        rows.append(r)
        if i % 500 == 0:
            print(f"  {i} files listed ...", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with open(Path(a.out) / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "size"]); w.writerows(rows)
    total = sum(s for _, s in rows)
    print(f"\n{len(rows)} files, {total / 1e9:.1f} GB -> {a.out}/manifest.csv")
    by_ext, by_dir = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for p, s in rows:
        by_ext[Path(p).suffix.lower() or "(none)"][0] += 1; by_ext[Path(p).suffix.lower() or "(none)"][1] += s
        key = "/".join(p.strip("/").split("/")[:a.depth])
        by_dir[key][0] += 1; by_dir[key][1] += s
    print("by type:", {k: f"{n} files, {s / 1e9:.1f} GB" for k, (n, s) in sorted(by_ext.items(), key=lambda x: -x[1][1])})
    print(f"by folder (depth {a.depth}):")
    for k, (n, s) in sorted(by_dir.items()):
        print(f"  {k:60s} {n:6d} files  {s / 1e9:7.2f} GB")
    print("first paths:", *[p for p, _ in rows[:8]], sep="\n  ")


def cmd_labels(a):
    rows = load_manifest(a.out)
    out = Path(a.out) / "labels"
    loose = select(rows, a.include, a.exclude)
    loose = [(p, s) for p, s in loose if re.search(LABEL_RE, p) and not p.lower().endswith(".zip")]
    zips = select(rows, a.include, a.exclude, ".zip")
    print(f"{len(zips)} zips to read labels from, {len(loose)} loose label files")
    if zips:   # partial reads need REST; a zip's last 22 bytes are its end-of-archive record ("PK\x05\x06")
        p, s = zips[0]
        try:
            r = FTPRange(lambda: connect(a.tls), p, s); r.seek(-22, 2); tail = r.read(22)
        except ftplib.Error as e:
            tail = str(e).encode()
        if not tail.startswith(b"PK\x05\x06"):
            sys.exit(f"the server does not support partial reads (got {tail[:40]!r}), so labels can't be read without "
                     "downloading whole zips; use the fetch command on a few zips instead")
        print("partial reads work")
    for p, s in loose:
        dst = out / p.strip("/")
        if not dst.exists():
            download(lambda: connect(a.tls), p, s, dst)
    for i, (p, s) in enumerate(zips, 1):
        dst = out / p.strip("/")[:-4]
        if (dst / ".done").exists():
            continue
        t0 = time.time()
        with zipfile.ZipFile(io.BufferedReader(FTPRange(lambda: connect(a.tls), p, s), buffer_size=1 << 18)) as z:
            members = [m for m in z.namelist() if re.search(a.members, m) and not m.endswith("/")]
            for m in members:
                target = dst / m
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(m))
        dst.mkdir(parents=True, exist_ok=True); (dst / ".done").write_text(json.dumps(members))
        print(f"[{i}/{len(zips)}] {p}: {len(members)} files ({time.time() - t0:.0f} s)", flush=True)
    print(f"labels in {out}")


def cmd_fetch(a):
    rows = select(load_manifest(a.out), a.include, a.exclude, ".zip")
    vids, work = Path(a.out) / "videos", Path(a.work)
    todo = [(p, s) for p, s in rows if not (vids / p.strip("/")[:-4] / "done.json").exists()]
    todo = todo[:a.limit] if a.limit else todo
    print(f"selected {len(rows)} zips ({sum(s for _, s in rows) / 1e9:.1f} GB), already done {len(rows) - len(todo) if not a.limit else '-'},"
          f" this run: {len(todo)} zips, {sum(s for _, s in todo) / 1e9:.1f} GB")
    if a.dry_run:
        print(*[f"  {p}  {s / 1e9:.2f} GB" for p, s in todo[:50]], sep="\n")
        return
    work.mkdir(parents=True, exist_ok=True)
    for i, (p, s) in enumerate(todo, 1):
        free = shutil.disk_usage(work).free
        if free < 3 * s:
            sys.exit(f"only {free / 1e9:.0f} GB free in {work}; need ~{3 * s / 1e9:.0f} GB for {p}")
        local = work / Path(p).name
        t0 = time.time()
        download(lambda: connect(a.tls), p, s, local)
        dl = time.time() - t0
        stats = process_zip(local, vids / p.strip("/")[:-4], work, a.crf, a.fps)
        local.unlink()
        stats.update(remote=p, download_s=round(dl), download_mb_s=round(s / 1e6 / max(dl, 1e-3), 1),
                     process_s=round(time.time() - t0 - dl))
        (vids / p.strip("/")[:-4] / "done.json").write_text(json.dumps(stats, indent=1))
        kept = sum(v["mb"] for v in stats["videos"].values())
        print(f"[{i}/{len(todo)}] {p}: {s / 1e9:.2f} GB -> {kept:.0f} MB video, {stats['download_mb_s']} MB/s, "
              f"process {stats['process_s']} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["list", "labels", "fetch"])
    ap.add_argument("--out", required=True, help="output folder (on Drive)")
    ap.add_argument("--tls", action="store_true", help="use FTP over TLS (if the plain login is refused)")
    ap.add_argument("--root", help="list: remote folder to start from (default: the login folder)")
    ap.add_argument("--depth", type=int, default=2, help="list: folder depth for the size summary")
    ap.add_argument("--include", help="regex on the remote path, e.g. 'group1/' or 'experiment(1|3)/'")
    ap.add_argument("--exclude", help="regex on the remote path, e.g. 'general_cam'")
    ap.add_argument("--members", default=LABEL_RE, help="labels: regex for files to pull out of each zip")
    ap.add_argument("--work", default="/content/dipser_work", help="fetch: local scratch folder (not Drive)")
    ap.add_argument("--limit", type=int, help="fetch: at most N zips this run")
    ap.add_argument("--crf", type=int, default=18, help="fetch: H.264 quality (lower = better, 18 ~ visually lossless)")
    ap.add_argument("--fps", type=float, default=10, help="fetch: frame rate written into the mp4")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    {"list": cmd_list, "labels": cmd_labels, "fetch": cmd_fetch}[a.command](a)


if __name__ == "__main__":
    main()
