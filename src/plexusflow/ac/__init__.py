"""AC (pulsatile) field on the carved network.

Modules:
  - ``field``       — Option-A harmonic field: tile time gauge ψ by angular synchronisation (coherent, k·ψ),
                       AC winner per edge, Ĥ_k = (|Q'|/|a0|) e^{−ikψ} H_k,raw.  No conservation imposed.
                       Per winner: own mean a0 and per-beat phase / pulsatility uncertainty; ``supported``
                       = the measurement-supported fit set (|Q'|/|a0| ≤ 5).

Current method: ``plexusflow.ac.pulse`` (driver ``scripts/pulse_field.py``); ``ac.field`` is the earlier synchronisation.  The network model and wall-property inference built on this field live in
``plexusmech.ac``.
"""
