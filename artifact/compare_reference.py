#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

EXACT=("candidate frames","application frames","decision frames")
NUMERIC=(
    "mean PSNR accuracy","mean SSIM accuracy","mean CR accuracy",
    "selected mean PSNR accuracy","selected mean SSIM accuracy","selected mean CR accuracy",
    "compressor + EB correct","compressor correct","EB correct",
    "selection coverage","evaluation coverage","mean relative joint-score gap",
)

def value(text:str,label:str)->float|None:
    match=re.search(rf"^{re.escape(label)}:\s*([\d.eE+-]+)",text,re.MULTILINE|re.IGNORECASE)
    return float(match.group(1)) if match else None

def main()->None:
    parser=argparse.ArgumentParser(description="Compare deterministic PILOT metrics; runtime is intentionally ignored.")
    parser.add_argument("candidate",type=Path,help="New experiment directory")
    parser.add_argument("reference",type=Path,help="Committed reference experiment directory")
    parser.add_argument("--atol",type=float,default=1e-5,help="Absolute tolerance for reported numerical metrics")
    args=parser.parse_args()
    candidate=(args.candidate/"run_summary.log").read_text();reference=(args.reference/"run_summary.log").read_text()
    failures=[]
    for label in EXACT:
        new=value(candidate,label);old=value(reference,label)
        if new is None or old is None:failures.append(f"{label}: missing")
        elif int(new)!=int(old):failures.append(f"{label}: candidate={int(new)}, reference={int(old)}")
        else:print(f"[MATCH] {label}: {int(new)}")
    for label in NUMERIC:
        new=value(candidate,label);old=value(reference,label)
        if new is None and old is None:continue
        if new is None or old is None:failures.append(f"{label}: present in only one log")
        elif not math.isclose(new,old,rel_tol=0.0,abs_tol=args.atol):failures.append(f"{label}: candidate={new:.8g}, reference={old:.8g}")
        else:print(f"[MATCH] {label}: {new:.6f}")
    if failures:raise SystemExit("Reference comparison FAILED:\n- "+"\n- ".join(failures))
    print("Reference comparison PASSED (runtime fields intentionally excluded).")

if __name__=="__main__":main()

