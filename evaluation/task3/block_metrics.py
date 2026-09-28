"""Blockade opportunities and adherence; absent telemetry is not a zero."""

BLOCK_PAIRS = (
    ("block_target", "block_target_opportunities", "block_target_followed"),
    ("block_local_wait", "block_local_wait_opportunities", "block_local_wait_followed"),
)
BLOCK_METRICS = ("tactical_block_diagnostic_rounds",) + tuple(
    name for prefix, _, _ in BLOCK_PAIRS for name in (
        f"total_{prefix}_opportunities", f"total_{prefix}_followed",
        f"mean_{prefix}_opportunities_per_round", f"{prefix}_follow_rate"))


def summarize_block_diagnostics(rows):
    measured = [row for row in rows if row.get("tactical_block_observations", 0)]
    result = {"tactical_block_diagnostic_rounds": len(measured)}
    for prefix, opportunity_name, followed_name in BLOCK_PAIRS:
        opportunities = followed = None
        if measured:
            for row in measured:
                n, k = row.get(opportunity_name, 0), row.get(followed_name, 0)
                if (type(n) is not int or type(k) is not int or not 0 <= k <= n):
                    raise ValueError(f"Invalid tactical counts: {opportunity_name}, {followed_name}")
            opportunities = sum(row.get(opportunity_name, 0) for row in measured)
            followed = sum(row.get(followed_name, 0) for row in measured)
        result.update({
            f"total_{prefix}_opportunities": opportunities,
            f"total_{prefix}_followed": followed,
            f"mean_{prefix}_opportunities_per_round": (
                opportunities / len(measured) if measured else None),
            f"{prefix}_follow_rate": followed / opportunities if opportunities else None,
        })
    return result
