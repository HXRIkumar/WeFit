"""Training-derived schedules, candidate set and validation-only selection rules."""

METHODS = (
    "frozen",
    "last1",
    "last2",
    "last4",
    "last12",
    "later8",
    "later16",
    "reuse4",
    "reuse12",
    "drop2",
    "student2",
    "drop6",
    "student6",
    "drop12",
    "student12",
)
TEMPORAL = METHODS[1:9]
STRUCTURAL = ("student2", "student6", "student12")
GATES = {
    "strict": {
        "mean_ssim": 0.95,
        "min_ssim": 0.90,
        "garment_mae_delta": 1.0,
        "outside_mae_delta": 0.5,
    },
    "exploratory": {
        "mean_ssim": 0.80,
        "min_ssim": 0.70,
        "garment_mae_delta": 3.0,
        "outside_mae_delta": 1.0,
    },
}


def plan(method, training):
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}")
    if method == "frozen":
        return [[], [], []]
    n = int("".join(c for c in method if c.isdigit()))
    if method.startswith(("student", "drop")):
        return [training["structural_order"][:n] for _ in range(3)]
    orders = training["step_orders"]
    return [[], orders[1][:n] if method.startswith("later") else [], orders[2][:n]]


def choose(summary, candidates, structural):
    baseline = summary["frozen"]
    for tier, gate in GATES.items():
        eligible = [
            name
            for name in candidates
            if summary[name]["mean_ssim"] >= gate["mean_ssim"]
            and summary[name]["min_ssim"] >= gate["min_ssim"]
            and summary[name]["garment_mae"]
            <= baseline["garment_mae"] + gate["garment_mae_delta"]
            and summary[name]["outside_mae"]
            <= baseline["outside_mae"] + gate["outside_mae_delta"]
        ]
        if eligible:
            if structural:
                return max(eligible, key=lambda m: int(m[7:])), tier
            return min(eligible, key=lambda m: summary[m]["median_seconds"]), tier
    return max(
        candidates, key=lambda m: summary[m]["mean_ssim"]
    ), "highest similarity; missed both gates"
