#!/usr/bin/env python3
"""Bump the package version in the single source of truth.

The version lives only in ``src/exoplanet_analysis/__init__.py`` (pyproject.toml
reads it dynamically). This helper updates it and prints the git commands to
commit, tag, and push the release.

Usage
-----
    python tools/bump_version.py 1.4.2
    python tools/bump_version.py patch    # 1.4.1 -> 1.4.2
    python tools/bump_version.py minor    # 1.4.1 -> 1.5.0
    python tools/bump_version.py major    # 1.4.1 -> 2.0.0
"""
import re
import sys
import os

INIT = os.path.join(os.path.dirname(__file__), "..", "src", "exoplanet_analysis", "__init__.py")
INIT = os.path.normpath(INIT)


def read_version(text):
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
    if not m:
        raise SystemExit("Could not find __version__ in " + INIT)
    return m.group(1)


def bump(current, part):
    major, minor, patch = (int(x) for x in current.split("."))
    if part == "major":
        return "{}.0.0".format(major + 1)
    if part == "minor":
        return "{}.{}.0".format(major, minor + 1)
    if part == "patch":
        return "{}.{}.{}".format(major, minor, patch + 1)
    # explicit version string
    if re.fullmatch(r"\d+\.\d+\.\d+", part):
        return part
    raise SystemExit("Argument must be 'major', 'minor', 'patch', or X.Y.Z")


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    text = open(INIT).read()
    current = read_version(text)
    new = bump(current, sys.argv[1])

    new_text = re.sub(r'(__version__\s*=\s*["\'])[^"\']+(["\'])',
                      r'\g<1>' + new + r'\g<2>', text)
    open(INIT, "w").write(new_text)

    print("Version bumped: {} -> {}".format(current, new))
    print()
    print("Next steps:")
    print("  git add -A")
    print('  git commit -m "Release v{}"'.format(new))
    print("  git tag v{}".format(new))
    print("  git push && git push --tags")


if __name__ == "__main__":
    main()
