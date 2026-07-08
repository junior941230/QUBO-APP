"""CHB-MIT channel normalization helpers.

The dataset contains both bipolar recordings and a small number of recordings
stored against a common reference.  This module builds a deterministic plan
for producing the same 18 bipolar channels from either representation.
"""

CANONICAL_CHB_CHANNELS = (
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FZ-CZ", "CZ-PZ",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
)


_BIPOLAR_ALIASES = {
    # Some CHB-MIT files contain two T8-P8 channels.  The "-0" channel is
    # the one in the standard temporal chain; "-1" belongs to an auxiliary
    # chain and must not silently replace it.
    "T8-P8": ("T8-P8", "T8-P8-0"),
}

_ELECTRODE_ALIASES = {
    # chb12_28 and chb12_29 spell O1 with a zero.
    "O1": ("O1", "01"),
}


def _first_present(candidates, channel_names):
    return next((name for name in candidates if name in channel_names), None)


def _find_electrode_source(electrode, channel_names):
    aliases = _ELECTRODE_ALIASES.get(electrode, (electrode,))
    candidates = []
    for alias in aliases:
        candidates.extend((alias, f"{alias}-CS2"))
    return _first_present(candidates, channel_names)


def build_channel_plan(channel_names):
    """Return instructions for constructing the canonical bipolar montage.

    Each returned item is either ``("direct", source)`` or
    ``("difference", anode_source, cathode_source)``.  Direct bipolar
    channels are preferred.  Missing bipolar channels are reconstructed from
    electrodes recorded against a common reference.
    """
    available = set(channel_names)
    plan = []
    missing = []

    for canonical in CANONICAL_CHB_CHANNELS:
        direct = _first_present(
            _BIPOLAR_ALIASES.get(canonical, (canonical,)),
            available,
        )
        if direct is not None:
            plan.append(("direct", direct))
            continue

        anode, cathode = canonical.split("-", 1)
        anode_source = _find_electrode_source(anode, available)
        cathode_source = _find_electrode_source(cathode, available)
        if anode_source is None or cathode_source is None:
            missing.append(canonical)
            continue
        plan.append(("difference", anode_source, cathode_source))

    if missing:
        raise ValueError(
            "Cannot construct canonical CHB montage; missing channels: "
            + ", ".join(missing)
        )
    return plan
