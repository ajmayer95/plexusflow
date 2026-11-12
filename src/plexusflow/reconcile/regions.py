"""Graph region classes used by the reconstruction (edge tags, set in the workbench viewer).

Two kinds of region are removed from the reconstruction network, with different physics:

  open        ``open_region=True`` — blood enters or leaves the network through it (the high-flux zones beside the dorsal
              aortae).  Every node where the remaining network meets an open region is a free boundary node and its
              exchange is counted as inflow (arterial injection).  Legacy key: ``radial_uncertain`` (read; also written alongside ``open_region``
              for plexusmech).
  topology    ``topology_uncertain=True`` — vessels exist and mass is conserved, but the graph there does not represent
              them (unresolvable marginal band, sheet-like flow, untraceable dense plexus).  Each connected topology
              region is a mass-conserving black box: the cut nodes around it exchange freely with each other, the net
              exchange of the region is zero, and its internal routing is unknown.

An edge carries at most one class; tagging an edge with one class clears the other.
"""
from __future__ import annotations

OPEN, TOPOLOGY = "open", "topology"
OPEN_KEY, TOPOLOGY_KEY, LEGACY_OPEN_KEY = "open_region", "topology_uncertain", "radial_uncertain"
LABELS = {OPEN: "Open boundary (inflow / outflow)", TOPOLOGY: "Topology uncertain (conserving)"}
COLORS = {OPEN: "#ff8800", TOPOLOGY: "#00d0ff"}


def region_class(d):
    """Class of an edge-attribute dict: 'open', 'topology' or None."""
    if d.get(TOPOLOGY_KEY):
        return TOPOLOGY
    if d.get(OPEN_KEY) or d.get(LEGACY_OPEN_KEY):
        return OPEN
    return None


def is_open(d):
    return region_class(d) == OPEN


def is_topology(d):
    return region_class(d) == TOPOLOGY


def set_region(d, cls):
    """Set the class of an edge-attribute dict in place (None clears)."""
    for k in (OPEN_KEY, TOPOLOGY_KEY, LEGACY_OPEN_KEY):
        d.pop(k, None)
    if cls == OPEN:
        d[OPEN_KEY] = True
        d[LEGACY_OPEN_KEY] = True          # still read by plexusmech's carved simulation
    elif cls == TOPOLOGY:
        d[TOPOLOGY_KEY] = True
    elif cls is not None:
        raise ValueError(f"unknown region class {cls!r}")


def counts(G):
    """{'open': n, 'topology': n} tagged edges in G."""
    out = {OPEN: 0, TOPOLOGY: 0}
    for _, _, d in G.edges(data=True):
        c = region_class(d)
        if c:
            out[c] += 1
    return out
