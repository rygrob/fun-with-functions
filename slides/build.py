"""Copy the deck into slides/dist/, ready to host as static files.

The deck loads reveal.js, KaTeX and Inter straight from node_modules, which is
fine locally but hosts tend to skip that name. So dist/ carries only the files
the deck uses, with each library's license, in vendor/, and the paths rewritten.

Usage:
    python3 slides/build.py
"""

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
VENDOR = {
    "reveal.js": ["dist", "LICENSE"],
    "katex": ["dist", "LICENSE"],
    "@fontsource-variable/inter": ["index.css", "files", "LICENSE"],
}


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def main():
    shutil.rmtree(DIST, ignore_errors=True)
    for name in ("index.html", "css", "js", "data", "img"):
        copy(ROOT / name, DIST / name)
    for package, names in VENDOR.items():
        for name in names:
            copy(ROOT / "node_modules" / package / name, DIST / "vendor" / package / name)
    for path in [DIST / "index.html", *(DIST / "js").rglob("*.js")]:
        path.write_text(re.sub(r"node_modules/", "vendor/", path.read_text()))
    files = [p for p in DIST.rglob("*") if p.is_file()]
    print(f"Wrote {DIST} ({len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
