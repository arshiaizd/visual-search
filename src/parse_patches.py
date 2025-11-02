import re
from typing import Set, Tuple

PAIR = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")

def parse_patch_pairs(text: str) -> Set[Tuple[int,int]]:
    """
    Expect output like: (r,c) (1,2) (3,4) or a list [(1,2), (3,4)].
    Returns a set of (r,c).
    """
    pairs = set()
    for m in PAIR.finditer(text):
        r = int(m.group(1)); c = int(m.group(2))
        pairs.add((r,c))
    return pairs
