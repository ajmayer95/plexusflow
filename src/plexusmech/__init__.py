"""plexusmech — mechanics and modelling on top of plexusflow reconstructions.

plexusflow (paper 1) turns videos into a measured, flow-conserving vessel network: per-tile measurements, the
DC reconciliation, the tile-aligned AC field and the plain Poiseuille comparison.  plexusmech (paper 2) asks what
wall mechanics explain that field: the distributed two-port (transmission-line) network model, boundary forcing,
the (D, η) likelihood and the wall-property inference.

Dependency rule: plexusmech imports plexusflow, never the other way round (``tests/test_package_boundary.py``).
New work should consume published releases through ``plexusflow.release.load_release``.

Sub-packages:
  - ``ac``         — two-port network model, von Mises / Student-t likelihood, group boundary-forcing (D, η) estimate
  - ``tl``         — transmission-line solver (time-harmonic, intact / carved simulations) and boundary helpers
  - ``inference``  — legacy carved-simulation state and boundary forcing (reads the legacy normalized fields)
  - ``viewer``     — backend for the viewer's simulation tabs
"""
__version__ = "0.1.0"
