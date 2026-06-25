# Mega-Heracross
**ISRO Bharatiya Antariksh Hackathon 2026 — Problem Statement 4**
Route Resilience: Occlusion-Robust Road Extraction & Graph-Theoretic Criticality Analysis

---

## Architecture

```
mega-heracross/
├── shared/
│   ├── schema.py      ← LOCKED contract dataclasses (do not modify)
│   ├── config.py      ← LOCKED constants (do not modify)
│   └── eval.py        ← Shared evaluation — all parts contribute here
│
├── part_a_vision/     ← Part A: Segmentation engine (not this repo)
│   └── outputs/
│       ├── road_mask.npy
│       └── meta.json
│
├── part_b_skeleton/   ← Part B: Skeletonization & Healing (THIS MODULE)
│   ├── run.py         ← ENTRYPOINT — run this
│   ├── outputs/
│   │   └── graph.json ← consumed by Part C
│   └── tests/
│
└── part_c_resilience/ ← Part C: Resilience & Visualization (not this repo)
    └── outputs/
        └── disaster_heatmap.html
```

## Quick Start — Part B

```bash
# From repo root
python -m part_b_skeleton.run

# Or directly
cd mega-heracross
python part_b_skeleton/run.py
```

## Contract

| File | Direction | Owner |
|------|-----------|-------|
| `part_a_vision/outputs/road_mask.npy` | A → B | Part A |
| `part_a_vision/outputs/meta.json` | A → B | Part A |
| `part_b_skeleton/outputs/graph.json` | B → C | Part B |

CRS: always `EPSG:4326`. Never break this.

## Phase Tracker

| Phase | Title | Status |
|-------|-------|--------|
| 01 | Repo scaffold & contract wiring | ✓ Done |
| 02 | Contract validation in shared/eval.py | ○ Next |
| 03 | Synthetic mask loader + geo-transform | ○ |
| 04 | Zhang-Suen skeletonization | ○ |
| 05 | sknw graph extraction + RoadGraph emission | ○ |
| ... | ... | ○ |

## Test Tile
Koramangala, Bengaluru: `(77.6101, 12.9177, 77.6401, 12.9377)`
