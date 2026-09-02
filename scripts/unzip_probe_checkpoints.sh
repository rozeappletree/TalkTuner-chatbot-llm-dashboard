#!/usr/bin/env bash
# Unzips every zip archive in data/probe_checkpoints/ into a sibling folder of
# the same name (minus .zip). Uses python3's zipfile module since `unzip` may
# not be installed. The extracted folders are already covered in .gitignore.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINTS_DIR="$REPO_ROOT/data/probe_checkpoints"

python3 - "$CHECKPOINTS_DIR" <<'EOF'
import sys, zipfile, os, glob

checkpoints_dir = sys.argv[1]
os.chdir(checkpoints_dir)

for f in sorted(glob.glob("*.zip")):
    dirname = f[:-4]
    with zipfile.ZipFile(f) as z:
        names = z.namelist()
        # If every entry is already nested under a top-level folder matching
        # dirname, extract next to the zip so we don't end up with
        # dirname/dirname/... Otherwise extract into dirname as before.
        if names and all(n.startswith(dirname + "/") for n in names):
            target = "."
        else:
            target = dirname
            os.makedirs(target, exist_ok=True)
        z.extractall(target)
    print(f"unzipped {f} -> {dirname}")
EOF
