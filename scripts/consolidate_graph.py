"""Consolidate cluttered stretches of the stage graph (plexusflow.edit.consolidate): fill pillar holes in the mask, redraw
speck loops and split junctions as single vessels / single junctions, leave regions whose redraw fails the checks for
manual review.

Writes through the stage patch log (plexusflow.edit.patches, graph/patches/): the change is recorded as a reversible
patch of kind 'consolidate' (revert it from the workbench Refine tab or PatchLog.revert), and the stage mask gets the
filled pillar holes (journaled in mask/edits; --keep-mask leaves mask.tif unchanged).  batch_profile_inputs/ is copied
(suffix .pre_consolidate_<date>).  Records G.graph['consolidation'], writes graph/consolidation_review.json (regions left
for manual review + the old→new edge map) and mask/mask_consolidated.tif (the mask the paths were drawn on).
Refuses to run twice.
Recordings: redrawn vessels lose theirs (geometry changed) unless --keep-recordings; --clear-all-recordings drops every
recording for a full re-measure.  Afterwards: `plexusflow analyze STAGE` (--skip-analyzed for the redrawn vessels only),
then scripts/run_stage.sh.
Usage: python scripts/consolidate_graph.py [--dry-run] [--clear-all-recordings | --keep-recordings] [--keep-mask]
"""
import sys, json, pickle, shutil, argparse, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import tifffile
from plexusflow.compat import sidecar_dir
import plexusflow.paths as paths
from plexusflow.edit.consolidate import consolidate, LT_MAX, SLIVER, K_SPAN


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--clear-all-recordings", action="store_true", help="drop every recording (full re-measure follows)")
    g.add_argument("--keep-recordings", action="store_true", help="keep recordings on redrawn vessels (not recommended)")
    ap.add_argument("--keep-mask", action="store_true", help="leave mask/mask.tif unchanged (a later region re-vectorisation "
                    "would then bring the filled holes back)")
    a = ap.parse_args()
    G = paths.load_graph()
    if "consolidation" in G.graph:
        sys.exit(f"graph already consolidated ({G.graph['consolidation'].get('date')}); restore the backup to redo it")
    m = tifffile.imread(str(paths.STAGE / "mask" / "mask.tif")) > 127
    H, emap, rep, mf = consolidate(G, m, log=lambda t: print(t, flush=True))
    print(f"holes filled {rep['holes_filled']}/{rep['holes_total']};  {G.number_of_nodes()}/{G.number_of_edges()} → "
          f"{H.number_of_nodes()}/{H.number_of_edges()} nodes/edges;  one-vessel {rep['one_vessel']}, junction {rep['junction']}, "
          f"left for review {len(rep['review'])}, too large {rep['too_large']}, skipped {rep['skipped']};  "
          f"recordings on internal strands dropped {rep['dropped_recordings']}")
    why = {}
    for r in rep["review"]:
        why[r["reason"]] = why.get(r["reason"], 0) + 1
    print("  review reasons:", why)
    if a.dry_run:
        return
    n_cleared = 0
    for u, v, d in H.edges(data=True):
        if a.clear_all_recordings or (d.get("consolidated") and not a.keep_recordings):
            n_cleared += len(d.get("measurements_piv") or []); d["measurements_piv"] = []
            for k in [k for k in d if k.startswith("_h_") or k in ("harmonic_class", "_meas_dissipation")]:
                d.pop(k)
    date = datetime.date.today().strftime("%Y%m%d"); gp = paths.GRAPH; prof = sidecar_dir(paths.STAGE)
    if prof.exists():
        shutil.copytree(prof, str(prof) + f".pre_consolidate_{date}", dirs_exist_ok=True)
    H.graph["consolidation"] = dict(date=date, lt_max=LT_MAX, sliver=SLIVER, k_span=K_SPAN, holes_filled=rep["holes_filled"],
                                    one_vessel=rep["one_vessel"], junction=rep["junction"], review=len(rep["review"]),
                                    recordings_cleared=n_cleared, all_cleared=bool(a.clear_all_recordings), mask_updated=not a.keep_mask)
    from plexusflow.edit.patches import PatchLog
    mp = paths.STAGE / "mask" / "mask.tif"
    print("[consolidate] writing the patch (graph diff with full attributes; large when recordings are cleared) …", flush=True)
    entry = PatchLog(gp, mp).accept(G, H, kind="consolidate",
                                    note=f"pillar holes + split junctions ({rep['holes_filled']} holes, {rep['junction']} junctions, "
                                         f"{rep['one_vessel']} one-vessel)" + ("; all recordings cleared" if a.clear_all_recordings else ""),
                                    report={k: v for k, v in H.graph["consolidation"].items()},
                                    mask_before=None if a.keep_mask else m, mask_after=None if a.keep_mask else mf)
    (gp.parent / "consolidation_review.json").write_text(json.dumps(dict(
        summary=H.graph["consolidation"], patch=entry["id"], review=rep["review"], regions=rep["regions"],
        edge_map=[[list(map(int, o)), None if n is None else list(map(int, n))] for o, n in emap.items()]), indent=1))
    tifffile.imwrite(str(paths.STAGE / "mask" / "mask_consolidated.tif"), (mf * 255).astype(np.uint8), compression="zlib")
    print(f"wrote {gp.name} as patch {entry['id']} (revert: workbench Refine tab), {n_cleared} recordings cleared, "
          f"mask {'unchanged' if a.keep_mask else 'updated (holes filled, journaled)'}, review list graph/consolidation_review.json")


if __name__ == "__main__":
    main()
