"""Code provenance for runs where the code is staged without a .git directory (Unity: git archive of a clean commit).

code_commit() returns (commit, dirty). If STAGED_COMMIT.txt exists next to this file (written when the code is staged
with git archive), its content is the commit and the copy is clean by construction; otherwise git is asked directly.
"""
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def code_commit():
    staged = os.path.join(HERE, "STAGED_COMMIT.txt")
    if os.path.exists(staged):
        with open(staged) as f:
            return f.read().strip(), False
    try:
        commit = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", HERE, "status", "--porcelain", "."], capture_output=True, text=True).stdout.strip() != ""
        return (commit or "unknown"), dirty
    except Exception:
        return "unknown", True
