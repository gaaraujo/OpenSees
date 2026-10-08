# modal damping examples

Compare four modal-damping variants on three models:

| Case tag | Command |
|----------|---------|
| `modalDampingQ` | `modalDampingQ` (quick, RHS only) |
| `modalDamping` | `modalDamping -legacy` (full tangent, BandGeneral or FullGeneral) |
| `modalDampingW` | `modalDamping -woodbury` (Woodbury, BandGeneral) |
| `modalDamping` + FullGeneral | reference for Woodbury iteration/response checks |

| Folder | Model |
|--------|--------|
| [four_story/](four_story/) | 4-story shear (Scott 2019) |
| [forty_story/](forty_story/) | 40-story shear |
| [two_story/](two_story/) | 2-story steel MRF (Kolay & Ricles) |

From any example folder: `python3 main.py` (or `OpenSees main.tcl`). Figures and results go to `figures/` and `results/`.
