"""Fail a build if a bundled executable or library cannot load on macOS 14 arm64."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import plistlib
import re
import subprocess
import sys


TARGET = (14, 0, 0)
# Thin 32/64-bit and universal 32/64-bit Mach-O, in both byte orders.
MACH_MAGIC = {bytes.fromhex(magic) for magic in (
    "feedface", "feedfacf", "cafebabe", "cafebabf",
    "cefaedfe", "cffaedfe", "bebafeca", "bfbafeca",
)}


def version(value):
    parts = tuple(int(x) for x in value.split("."))
    return (parts + (0, 0))[:3]


def binary_target(path):
    with path.open("rb") as f:
        if f.read(4) not in MACH_MAGIC:
            return None
    result = subprocess.run(["/usr/bin/otool", "-arch", "arm64", "-l", str(path)],
                            check=True, capture_output=True, text=True)
    for command in re.split(r"(?m)^Load command \d+\s*$", result.stdout):
        if re.search(r"(?m)^\s*cmd LC_BUILD_VERSION$", command):
            entry = re.search(r"(?m)^\s*minos ([\d.]+)\s*$", command)
        elif re.search(r"(?m)^\s*cmd LC_VERSION_MIN_MACOSX$", command):
            entry = re.search(r"(?m)^\s*version ([\d.]+)\s*$", command)
        else:
            continue
        if entry:
            return path, version(entry[1])
    raise ValueError(f"No arm64 macOS deployment target in {path}")


def main(app):
    with (app / "Contents/Info.plist").open("rb") as f:
        info = plistlib.load(f)
    minimum = info.get("LSMinimumSystemVersionByArchitecture", {}).get(
        "arm64", info.get("LSMinimumSystemVersion", "0"))
    if version(minimum) > TARGET:
        raise ValueError(f"Info.plist requires macOS {minimum}")
    paths = (p for p in app.rglob("*") if not p.is_symlink() and p.is_file())
    with ThreadPoolExecutor(max_workers=4) as pool:
        inspected = [result for result in pool.map(binary_target, paths) if result]
    if not inspected:
        raise ValueError(f"No Mach-O binaries found in {app}")
    issues = [f"{path.relative_to(app)} requires {'.'.join(map(str, minimum))}"
              for path, minimum in inspected if minimum > TARGET]
    if issues:
        raise ValueError("macOS 14.0 arm64 compatibility check failed:\n" + "\n".join(issues))
    print(f"Checked {len(inspected)} Mach-O deployment targets for macOS 14.0 arm64")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
