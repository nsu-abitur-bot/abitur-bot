"""Render a reviewed root-wrapper extension; never install it or change sudo."""

import argparse
import hashlib
from pathlib import Path

ORIGINAL_SHA256 = "deaf76c21681979ac96966b9b35ded30d4a1eb514c0b9f25236a627afaf22c4e"
MARKER = b"variant,action,*args=sys.argv[1:]\n"
DISPATCH = b"""if action == 'cli-run':
 if not args: die('Expected /evaluation/script.py and optional arguments')
 os.execv('/usr/bin/python3', [
  '/usr/bin/python3', '-I', '/usr/local/libexec/abitur-eval388-cli.py', variant, *args])
"""


def render(original: bytes) -> bytes:
    if hashlib.sha256(original).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("Wrapper changed: inspect it before preparing another patch")
    if original.count(MARKER) != 1:
        raise ValueError("Unexpected wrapper dispatch")
    return original.replace(MARKER, MARKER + DISPATCH)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = render(args.original.read_bytes())
    with args.output.open("xb") as output:
        output.write(result)


if __name__ == "__main__":
    main()
