"""Download the SCADA archives from Zenodo and check their md5 sums.

The file list is data/zenodo_files.tsv (record id, file name, size, md5),
taken from the current versions of the two datasets:
Kelmarsh doi:10.5281/zenodo.16807551 and Penmanshiel doi:10.5281/zenodo.16807304.
A file that already exists with the right md5 is not downloaded again.
"""
from __future__ import annotations

import hashlib
import urllib.parse
import urllib.request

import pandas as pd

from . import config as C


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    C.ensure_dirs()
    files = pd.read_csv(C.ROOT / "data" / "zenodo_files.tsv", sep="\t", dtype=str)
    for row in files.itertuples():
        out = C.RAW / row.file
        if out.exists() and md5(out) == row.md5:
            print(f"ok (already present) {row.file}")
            continue
        url = f"https://zenodo.org/records/{row.record}/files/{urllib.parse.quote(row.file)}?download=1"
        print(f"downloading {row.file} ({int(row.bytes) / 1e6:.0f} MB)", flush=True)
        urllib.request.urlretrieve(url, out)
        got = md5(out)
        if got != row.md5:
            raise RuntimeError(f"md5 mismatch for {row.file}: {got} != {row.md5}")
        print(f"ok {row.file}")


if __name__ == "__main__":
    main()
