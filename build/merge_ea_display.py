# -*- coding: utf-8 -*-
"""nets/toby2_level_cal.json for the next package: the shipped calibration (frozen_open_c/calibration.json, sha256 02472f1a...) with the
frozen E_A display fields (frozen_open_c/ea_display_fields.json, sha256 f238896e..., Yosha 2026-11-22) added per group under the key
"ea_display". Nothing else changes: removing every group's "ea_display" gives back the calibration exactly (checked here, as parsed JSON).
  python merge_ea_display.py OUT.json [--cal calibration.json] [--fields ea_display_fields.json]
Writes OUT.json (UTF-8, LF, json.dumps indent 1, no trailing newline) and prints its sha256."""
import copy
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAL_SHA = "02472f1ae54dc4bb4fe41b18c781a923c53024025a9a6a986d5b602b7cdebb80"
FIELDS_SHA = "f238896e432674be827d4984d1bd5a6d476cffc682cba936e6677ab91076721e"
KEYS = ("ea_stretch", "ea_coef_review", "ea_stretch_review", "ea_cap_high", "ea_cap_low")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def merge(cal, fields):
    if fields["model"] != cal["model"]:
        raise SystemExit(f"model {fields['model']} != {cal['model']}")
    if set(fields["groups"]) != set(cal["groups"]):
        raise SystemExit(f"groups {sorted(fields['groups'])} != {sorted(cal['groups'])}")
    out = copy.deepcopy(cal)
    for g, f in fields["groups"].items():
        if set(f) != set(KEYS) or len(f["ea_coef_review"]) != 5 or "ea_display" in out["groups"][g]:
            raise SystemExit(f"group {g}: unexpected fields {sorted(f)}")
        out["groups"][g]["ea_display"] = {k: f[k] for k in KEYS}
    back = copy.deepcopy(out)
    for g in back["groups"].values():
        del g["ea_display"]
    if back != cal:
        raise SystemExit("merge changed the calibration")
    return out


def main(a):
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d
    cp = opt("--cal", os.path.join(HERE, "frozen_open_c", "calibration.json"))
    fp = opt("--fields", os.path.join(HERE, "frozen_open_c", "ea_display_fields.json"))
    for p, want in ((cp, CAL_SHA), (fp, FIELDS_SHA)):
        if sha(p) != want:
            raise SystemExit(f"{p}: sha256 {sha(p)} != {want}")
    out = merge(json.load(open(cp, encoding="utf-8")), json.load(open(fp, encoding="utf-8")))
    with open(a[0], "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, indent=1))
    print("MERGED", a[0], sha(a[0]))


if __name__ == "__main__":
    main(sys.argv[1:])
