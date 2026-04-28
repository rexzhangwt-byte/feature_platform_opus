"""Synthesize mall-classification data:
  - people_count.json : key->list[48]  (24h workday + 24h weekend)
  - dwell_time.json   : key->list[48]
  - mall_label.json   : key->"商场"|"非商场"

The synthetic data is made so that "商场" sites have a noticeable
weekend boost in evening hours and longer dwell times — there's enough
signal that even a small AE+MLP can learn it.
"""
import json
import math
import random
from pathlib import Path

random.seed(0)
N_MALL, N_NON = 80, 80
OUT = Path(__file__).parent.parent / "scripts" / "mall_dataset"
OUT.mkdir(parents=True, exist_ok=True)


def _hours_curve(weekend: bool, is_mall: bool) -> list[float]:
    out = []
    for h in range(24):
        # baseline: bell around lunch & evening
        base = 30 + 60 * math.exp(-((h - 13) ** 2) / 14) \
                  + 80 * math.exp(-((h - 19) ** 2) / 8)
        if is_mall:
            base *= (1.4 if weekend else 0.9)
            base += 25 * math.exp(-((h - 15) ** 2) / 5) if weekend else 0
        else:
            base *= (0.7 if weekend else 1.1)
        base += random.gauss(0, 8)
        out.append(max(0.0, base))
    return out


def _dwell_curve(weekend: bool, is_mall: bool) -> list[float]:
    out = []
    for h in range(24):
        if is_mall:
            base = 35 + (20 if weekend else 5) + 15 * math.exp(-((h - 19) ** 2) / 9)
        else:
            base = 12 + 4 * math.exp(-((h - 8) ** 2) / 5)
        base += random.gauss(0, 3)
        out.append(max(0.0, base))
    return out


people, dwell, labels = {}, {}, {}
for i in range(N_MALL):
    k = f"mall_{i:03d}"
    people[k] = _hours_curve(False, True) + _hours_curve(True, True)
    dwell[k]  = _dwell_curve(False, True) + _dwell_curve(True, True)
    labels[k] = "商场"
for i in range(N_NON):
    k = f"site_{i:03d}"
    people[k] = _hours_curve(False, False) + _hours_curve(True, False)
    dwell[k]  = _dwell_curve(False, False) + _dwell_curve(True, False)
    labels[k] = "非商场"

# Add a few "rogue" keys to people but missing from labels to test intersection
for i in range(5):
    k = f"unknown_{i}"
    people[k] = _hours_curve(False, False) + _hours_curve(True, False)
    dwell[k]  = _dwell_curve(False, False) + _dwell_curve(True, False)


def write(p, obj):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    print(f"wrote {p}: {len(obj)} keys")


write(OUT / "people_count.json", people)
write(OUT / "dwell_time.json",   dwell)
write(OUT / "mall_label.json",   labels)
