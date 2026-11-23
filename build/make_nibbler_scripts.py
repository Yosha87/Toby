# -*- coding: utf-8 -*-
"""Writes Nibbler 'Custom scripts' (text files of `setoption` lines) so that Toby is set in three clicks: 1 the time control, 2 the rating of White, 3 the rating of Black.
Usage: python make_nibbler_scripts.py [rating ...]   (default levels: 1400 1600 1800 2000 2200 2300 2400 2500 2600 2700 2800)
Files go to ./nibbler_scripts/ ; copy them to Nibbler's scripts folder (Nibbler menu Engine > Custom scripts > Show scripts folder) and restart Nibbler.
  'Toby Classical' / 'Toby Rapid' / 'Toby Blitz'  : Rating Source FIDE and Time Control
  'Toby White <r>' : White Rating     'Toby Black <r>' : Black Rating (no digit after 'Toby': the names must not read as versions)
Give BOTH ratings (equal ratings = 'same strength on both sides'). If only one is set, the other colour uses the 'Rating' option (engines.json)."""
import sys
from pathlib import Path

ratings = [int(a) for a in sys.argv[1:]] or [1400, 1600, 1800, 2000, 2200, 2300, 2400, 2500, 2600, 2700, 2800]
out = Path(__file__).resolve().parent / "nibbler_scripts"
out.mkdir(exist_ok=True)
for old in out.glob("*.txt"):
    old.unlink()
for tc in ("Classical", "Rapid", "Blitz"):
    (out / f"Toby {tc}.txt").write_text(f"setoption name Rating Source value FIDE\nsetoption name Time Control value {tc}\n", encoding="utf-8")
for r in ratings:
    (out / f"Toby White {r}.txt").write_text(f"setoption name White Rating value {r}\n", encoding="utf-8")
    (out / f"Toby Black {r}.txt").write_text(f"setoption name Black Rating value {r}\n", encoding="utf-8")
print(f"{len(list(out.glob('*.txt')))} scripts written to {out}")
