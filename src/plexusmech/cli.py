"""plexusmech command-line interface (model side; reads plexusflow stage folders).

    plexusmech simulate <STAGE_ROOT>   # intact two-pass transmission-line simulation (LOCKED 2026-09-30 recipe)
"""
from __future__ import annotations

from pathlib import Path
import typer

from plexusflow.config import StageConfig

app = typer.Typer(add_completion=False, help="Vascular mechanics models on plexusflow reconstructions.")


@app.command()
def simulate(stage_root: Path = typer.Argument(...),
              src: Path = typer.Option(None,
                    help='Source .gpickle (default: graph/graph_vessels_piv_normalized.gpickle)'),
              out: Path = typer.Option(None,
                    help='Output .gpickle (default: same as src, sim fields added in place)'),
              D: float = typer.Option(3.5e-4,
                    help='Areal distensibility (1/Pa). Default 3.5e-4.'),
              eta: float = typer.Option(0.0,
                    help='Constant-loss-tangent wall η. D_eff = D · (1 − iη).'),
              da_dc: float = typer.Option(1.593,
                    help='Symmetric DA anchor DC injection (nL/s).'),
              da_h1_amp: float = typer.Option(0.512,
                    help='Symmetric DA anchor |H1| injection (nL/s).'),
              da_h1_phase_deg: float = typer.Option(139.6,
                    help='DA H1 phase (degrees) — cardiac pulse phase reference.'),
              ) -> None:
    """Run the intact two-pass sim (LOCKED 2026-09-30 recipe).

    Pass 1: sink_pressure_bc=0 at V (network+DA determine drainage DC).
    Pass 2: V injection = ratio × Pass-1 V DC with cross-tile v_ratios
    from `_h_*_norm` fields.

    Same code path as the viewer's Simulate button — both call
    `plexusmech.tl.transmission_line.solve_transmission_line`.
    Writes `_h_{amp,phase,Z}_H*_sim_intact` on every edge."""
    import math, pickle
    import numpy as np
    from .tl import transmission_line as tl
    cfg = StageConfig.load(stage_root)
    src = src or cfg.graph_normalized_path
    out = out or src
    if not src.exists():
        raise FileNotFoundError(f"Source graph not found: {src}\n"
                                f"Run `plexusflow legacy-normalize` first.")
    print(f"[simulate] loading  {src.name}   ({src.stat().st_size/1024/1024:.0f} MB)")
    G = pickle.loads(src.read_bytes())
    print(f"[simulate] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    A = sorted([n for n,d in G.nodes(data=True)
                if d.get('boundary_type')=='source'],
               key=lambda n: G.nodes[n]['x'])
    V = sorted([n for n,d in G.nodes(data=True)
                if d.get('boundary_type')=='sink'],
               key=lambda n: G.nodes[n]['x'])
    if len(A) < 1 or len(V) < 1:
        raise RuntimeError(f"Need source + sink anchors; got A={A} V={V}")
    print(f"[simulate] anchors: DA={A}  V={V}")

    N_HARM = 3
    f0 = float(G.graph.get('f0_canonical_hz', 2.467))
    DA_PHASE = math.radians(da_h1_phase_deg)

    # η patch on _vessel_admittance
    _orig_va = tl._vessel_admittance
    def _patched(R_m, L_m, omega, mu=tl.MU_DEFAULT, rho=tl.RHO_BLOOD, D=None):
        return _orig_va(R_m, L_m, omega, mu, rho,
                          (D if D is not None else 1e-6) * (1.0 - 1j*eta))
    tl._vessel_admittance = _patched

    # Symmetric DA BCs
    bc_pass1 = {}
    for a in A:
        arr = np.zeros(N_HARM + 1, dtype=complex)
        arr[0] = float(da_dc)
        arr[1] = da_h1_amp * np.exp(1j*DA_PHASE)
        bc_pass1[a] = arr
    print(f"[simulate] Pass 1: sink_pressure_bc=0.0 at V, DAs Q-BC "
          f"(DC={da_dc:+.3f}, |H1|={da_h1_amp:.3f}∠{da_h1_phase_deg:+.1f}°)")
    r1 = tl.solve_transmission_line(
        G, D=D, n_harmonics=N_HARM, f0_hz=f0,
        bc_harmonics_override=bc_pass1,
        sink_pressure_bc=0.0, verbose=False)
    print(f"[simulate] Pass 1 done ({len(r1.edge_flows)} edge flows)")

    # v_ratios from cross-tile _h_*_norm (matches pipeline_outlet_agg)
    def _norm_meas(d):
        dc = d.get('_h_amp_DC_norm')
        if dc is None or not np.isfinite(dc) or abs(dc) < 1e-30: return None
        out = {'DC': float(dc)}
        for k in (1, 2, 3):
            a = d.get(f'_h_amp_H{k}_norm'); p = d.get(f'_h_phase_H{k}_norm')
            if a is None or p is None or not (np.isfinite(a) and np.isfinite(p)):
                continue
            out[f'H{k}'] = complex(float(a)*np.exp(1j*float(p)))
        return out if 'H1' in out else None
    def _stub_flow_dc(sim, node):
        nb = list(G.neighbors(node))[0]
        if (node, nb) in sim.edge_flows:
            return float(complex(sim.edge_flows[(node, nb)][0]).real)
        ef = sim.edge_flows[(nb, node)]
        return -float(complex(ef[0]).real)

    v_ratios = {}
    for v_anchor in V:
        for _, nb in G.edges(v_anchor):
            d = G.edges[v_anchor, nb]
            mm = _norm_meas(d)
            if mm is None: continue
            v_ratios[v_anchor] = {
                k: (mm.get(f'H{k}') / mm['DC']) if mm.get(f'H{k}') is not None
                    else complex(0)
                for k in (1, 2, 3)}
            break

    # Pass 2 BCs
    bc_pass2 = dict(bc_pass1)
    for v_anchor in V:
        v1_dc = _stub_flow_dc(r1, v_anchor)
        ratios = v_ratios.get(v_anchor, {})
        arr = np.zeros(N_HARM + 1, dtype=complex)
        arr[0] = v1_dc
        for k in (1, 2, 3):
            if k in ratios:
                arr[k] = v1_dc * ratios[k]
        bc_pass2[v_anchor] = arr
        print(f"[simulate] Pass 1 V DC at {v_anchor}: {v1_dc:+.4f}  →  "
              f"Pass 2 |H1|={abs(arr[1]):.3f}∠{math.degrees(np.angle(arr[1])):+.1f}°")

    print("[simulate] Pass 2: solving with full Q-BC set …")
    r2 = tl.solve_transmission_line(
        G, D=D, n_harmonics=N_HARM, f0_hz=f0,
        bc_harmonics_override=bc_pass2, verbose=False)
    tl._vessel_admittance = _orig_va

    # Write intact-sim fields onto every edge
    n_written = 0
    for (u, v), ef in r2.edge_flows.items():
        d = G.edges[u, v]
        for k in range(N_HARM + 1):
            z = complex(ef[k])
            suffix = 'DC' if k == 0 else f'H{k}'
            d[f'_h_Z_{suffix}_sim_intact']    = z
            d[f'_h_amp_{suffix}_sim_intact']  = float(abs(z))
            if k > 0:
                d[f'_h_phase_{suffix}_sim_intact'] = float(np.angle(z))
        n_written += 1
    print(f"[simulate] wrote _h_*_sim_intact on {n_written} edges")

    print(f"[simulate] writing  {out.name}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump(G, f)
    print(f"[simulate] wrote {out.stat().st_size/1024/1024:.0f} MB")
