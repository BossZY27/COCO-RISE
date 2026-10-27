"""One-way legacy environment migration. Never log values or overwrite RISE keys."""
import argparse
import os
import re
import tempfile
from pathlib import Path

PREFIX = "IRFA_"


def migrate(path: Path):
    if not path.exists():
        return 0
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    existing = set(re.findall(r"^(?:export\s+)?(RISE_[A-Z0-9_]+)\s*=", original, re.M))
    migrated = 0
    output = []
    for line in lines:
        match = re.match(r"(\s*(?:export\s+)?)(IRFA_[A-Z0-9_]+)(\s*=.*)", line)
        if match:
            key = "RISE_" + match[2][len(PREFIX):]
            if key == "RISE_LLM_MAIN":
                key = "RISE_ANSWER_MODEL"
            if key not in existing:
                output.append(match[1] + key + match[3] + ("\n" if line.endswith("\n") else ""))
                existing.add(key)
                migrated += 1
        else:
            output.append(line.replace(PREFIX, "RISE_"))
    content = "".join(output)
    if content != original:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            os.chmod(temporary, 0o600)
            handle.write(content)
        os.replace(temporary, path)
    return migrated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=Path(".env"))
    args = parser.parse_args()
    print(f"Migrated {migrate(args.path)} variable names; values withheld.")
