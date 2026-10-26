#!/usr/bin/env bash
# Apply the frozen parameter constants to a user-supplied upstream checkout.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mode=apply
dry_run=0
repo=.
repo_given=0

usage() {
    cat <<'HELP'
Usage: bash apply_params.sh [--dry-run] [--restore] [UPSTREAM_DIRECTORY]

Run from the upstream project root, or supply its directory explicitly.
  --dry-run  Validate and preview changes without writing files.
  --restore  Restore the original parameter values instead of applying tuning.
  --help     Show this help.

Requires Bash, Git and Python 3.8+. Does not download, build or run the model.
HELP
}

while (($#)); do
    case "$1" in
        --dry-run) dry_run=1 ;;
        --restore) mode=restore ;;
        -h|--help) usage; exit 0 ;;
        -*) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
        *)
            if ((repo_given)); then
                printf 'Only one upstream directory may be supplied.\n' >&2
                exit 2
            fi
            repo=$1
            repo_given=1
            ;;
    esac
    shift
done

for tool in python3 git; do
    command -v "$tool" >/dev/null || { printf 'Required tool missing: %s\n' "$tool" >&2; exit 2; }
done

python3 - "$SCRIPT_DIR/parameter_patch.json" "$repo" "$mode" "$dry_run" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile

if sys.version_info < (3, 8):
    raise SystemExit("ERROR: Python 3.8 or newer is required.")


def fail(message):
    raise SystemExit("ERROR: " + message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


manifest = json.loads(Path(sys.argv[1]).read_text())
root = Path(sys.argv[2]).resolve()
mode = sys.argv[3]
dry_run = sys.argv[4] == "1"
if not root.is_dir():
    fail("Upstream directory not found: " + str(root))

try:
    top = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        text=True, stderr=subprocess.PIPE,
    ).strip()
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
    ).strip()
except subprocess.CalledProcessError:
    fail("Use a Git checkout of the upstream project, not this recipe directory.")
if Path(top).resolve() != root:
    fail("Run from the upstream project root, not a subdirectory.")
if commit != manifest["upstream_commit"]:
    fail("Unsupported upstream commit: " + commit + ". Expected " + manifest["upstream_commit"])

patches = {}
for parameter in manifest["parameters"]:
    patches.setdefault(parameter["file"], []).extend(parameter["replacements"])

# Validate every upstream tracked file before preparing any writes. Only the
# three exact default/tuned source variants are accepted; other edits are refused.
originals = {}
prepared = {}
for relative, original_hash in manifest["original_sha256"].items():
    path = root / relative
    if any(part.is_symlink() for part in [path, *path.parents] if part != root):
        fail("Symlinked upstream path is unsupported: " + relative)
    if not path.is_file():
        fail("Missing upstream file: " + relative)
    data = path.read_bytes()
    digest = sha(data)
    tuned_hash = manifest["tuned_sha256"].get(relative)
    if digest == original_hash:
        state = "default"
    elif tuned_hash and digest == tuned_hash:
        state = "tuned"
    else:
        fail("Unexpected contents in " + relative + "; no files have been changed.")
    if relative not in patches:
        continue
    target = "default" if mode == "restore" else "tuned"
    if state == target:
        print(relative + ": already " + target)
        continue
    updated = data
    for patch in patches[relative]:
        before = patch["before"] if mode == "apply" else patch["after"]
        after = patch["after"] if mode == "apply" else patch["before"]
        old, new = before.encode("ascii"), after.encode("ascii")
        if updated.count(old) != 1:
            fail("Expected exactly one expression in " + relative + ": " + before)
        updated = updated.replace(old, new, 1)
    expected = original_hash if mode == "restore" else tuned_hash
    if sha(updated) != expected:
        fail("Patched hash mismatch in " + relative + "; no files have been changed.")
    originals[path] = data
    prepared[path] = updated
    print(relative + ": " + state + " -> " + target)

for parameter in manifest["parameters"]:
    before, after = parameter["default"], parameter["tuned"]
    if mode == "restore":
        before, after = after, before
    print("  " + parameter["name"] + ": " + str(before) + " -> " + str(after))

if dry_run:
    print("Dry run complete; no files written.")
    sys.exit(0)

# Stage all writes first, preserving file modes and original CRLF bytes.
# Roll back completed replacements if a write fails.
staged = {}
replaced = []
try:
    for path, data in prepared.items():
        fd, temporary = tempfile.mkstemp(prefix=".parameter_patch_", dir=path.parent)
        staged[path] = Path(temporary)
        with os.fdopen(fd, "wb") as output:
            output.write(data)
        os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
    for path in prepared:
        if path.read_bytes() != originals[path]:
            fail("File changed during patch preparation: " + str(path))
    for path, temporary in staged.items():
        os.replace(temporary, path)
        replaced.append(path)
except BaseException:
    for path in reversed(replaced):
        path.write_bytes(originals[path])
    raise
finally:
    for temporary in staged.values():
        temporary.unlink(missing_ok=True)

print("Done: " + str(len(prepared)) + " source files updated. Review with git diff before rebuilding.")
PY
