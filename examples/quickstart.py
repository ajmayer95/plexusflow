"""Quickstart: from a frozen release to a reconciled flow field and two plots.

    export PLEXUSFLOW_STAGE=/path/to/PlexusFlow21S_release         # the downloaded release (read-only)
    export PLEXUSFLOW_RESULTS=$HOME/plexusflow_out/results          # your own output folders
    export PLEXUSFLOW_FIGURES=$HOME/plexusflow_out/figures
    python examples/quickstart.py

Steps: load the measured graph → fit the reliability scale c on same-tile junction balance → reconcile the DC field on
the carved network (conservation at interior junctions, emergent boundary fluxes) → compare with a passive Poiseuille
network → plot.  Takes about a minute.
"""
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import tifffile

import plexusflow.paths as paths
from plexusflow.reconcile import measurement_table, fit_reliability, reconcile_dc

# 1. the measured graph: one node per junction/endpoint, one edge per vessel segment; each edge carries its per-tile
#    optical-flow measurements (G.edges[u, v]["measurements_piv"]), see graph/export/SCHEMA.md in the release
G = paths.load_graph()
print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# 2. profile-valid measurements and the reliability scale c = c_T + c_f (1 − f) + c_a (1 − align)
M = measurement_table(G)
model, junctions = fit_reliability(G, M)
print(f"reliability: c_f {model['c_f']:.2f}, c_a {model['c_a']:.2f}, from {model['n_junctions']} same-tile junctions")

# 3. reconcile: per edge the most reliable measurement, weighted least squares subject to conservation
D = reconcile_dc(G, model, M)
m = D["metrics"]
print(f"inflow {m['total_in']:.2f} nL/s (L {m['inj_L']:.2f} / R {m['inj_R']:.2f});  veins V1/V2 {m['V'][0]:+.2f}/{m['V'][1]:+.2f};  "
      f"passive-network corr {m['corr_lin']:.3f} (shuffled radii {D['shuffled_corr']:.3f})")
paths.RESULTS.mkdir(parents=True, exist_ok=True)
(paths.RESULTS / "quickstart_dc.pkl").write_bytes(pickle.dumps({k: v for k, v in D.items() if k != "dom"}))

# 4a. reconciled |Q'| on the mosaic
mos = tifffile.imread(paths.MOSAIC); mos = mos.mean(-1) if mos.ndim == 3 else mos; H = mos.shape[0]
segs = [np.column_stack([np.asarray(G.edges[e]["path"], float)[:, 0], H - np.asarray(G.edges[e]["path"], float)[:, 1]]) for e in D["edges"]]
q = np.abs(D["Qp"]); vmin, vmax = np.percentile(q[q > 0], [5, 99])
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(mos, cmap="gray", vmin=np.percentile(mos, 2), vmax=np.percentile(mos, 98))
lc = LineCollection(segs, array=np.log10(np.clip(q, vmin, vmax)), cmap="viridis", linewidths=np.where(D["imaged"], 1.2, 0.5))
ax.add_collection(lc); fig.colorbar(lc, ax=ax, shrink=0.6, label="log10 |Q'| (nL/s)"); ax.set_axis_off()
ax.set_title("Reconciled mean flow (thin lines: unmeasured, filled in by conservation)")
paths.FIGURES.mkdir(parents=True, exist_ok=True); fig.savefig(paths.FIGURES / "quickstart_flow_map.png", dpi=150, bbox_inches="tight")

# 4b. measured vs reconciled, reconciled vs passive network
im = D["imaged"]
fig, axs = plt.subplots(1, 2, figsize=(11, 5))
for ax, (x, y, xl, yl) in zip(axs, ((np.abs(D["Qm"][im]), np.abs(D["Qp"][im]), "measured |Q| (nL/s)", "reconciled |Q'|"),
                                    (np.abs(D["Qp"][im]), np.abs(D["Qs"][im]), "reconciled |Q'|", "passive network |Q_sim|"))):
    ax.loglog(x, y, ".", ms=2, alpha=0.4); lim = [1e-4, 3]; ax.plot(lim, lim, "k-", lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel(xl); ax.set_ylabel(yl)
axs[1].set_title(f"r = {m['corr_lin']:.2f} (log {m['corr_log']:.2f})")
fig.tight_layout(); fig.savefig(paths.FIGURES / "quickstart_scatter.png", dpi=150)
print(f"wrote {paths.FIGURES / 'quickstart_flow_map.png'}, {paths.FIGURES / 'quickstart_scatter.png'}, {paths.RESULTS / 'quickstart_dc.pkl'}")
