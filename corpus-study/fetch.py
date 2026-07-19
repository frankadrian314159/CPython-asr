"""Fetches the corpus: shallow-clones every repo in manifest.json into
./corpus/<org>__<repo>, and writes manifest.lock.json recording the exact
commit SHA cloned, for reproducibility -- mirrors
docs/cgo2027/corpus-study/src/asr_corpus/fetch.clj in the FOL repo.

Usage: python fetch.py [manifest.json] [corpus_dir]
"""

import json
import subprocess
import sys
from pathlib import Path


def clone_one(entry, corpus_dir):
    org_repo = entry["org_repo"]
    dest_name = org_repo.replace("/", "__")
    dest = corpus_dir / dest_name
    if dest.exists():
        print(f"  skip (already present): {org_repo}")
    else:
        print(f"  cloning {org_repo} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", entry["url"], str(dest)],
            check=True,
            capture_output=True,
        )
    sha = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {**entry, "sha": sha}


def main():
    manifest_path = Path(sys.argv[1] if len(sys.argv) > 1 else "manifest.json")
    corpus_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "corpus")
    corpus_dir.mkdir(exist_ok=True)

    manifest = json.loads(manifest_path.read_text())
    locked = []
    for entry in manifest["repos"]:
        locked.append(clone_one(entry, corpus_dir))

    lock_path = manifest_path.parent / "manifest.lock.json"
    lock_path.write_text(json.dumps({"repos": locked}, indent=2) + "\n")
    print(f"\nCloned {len(locked)} repos into {corpus_dir}/; wrote {lock_path}")


if __name__ == "__main__":
    main()
