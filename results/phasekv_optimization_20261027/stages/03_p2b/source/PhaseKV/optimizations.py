"""Explicit, serializable switches. Unvalidated GPU experiments stay opt-in."""
from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class OptimizationConfig:
    fused_quantize: bool = False
    reuse_workspace: bool = False
    paged_cold: bool = False
    page_slots: int = 32
    cold_group: int = 1
    specialized_4bit: bool = False
    gqa_reuse: bool = False
    hot_ring: bool = False

    def __post_init__(self):
        for name in ("fused_quantize", "reuse_workspace", "paged_cold",
                     "specialized_4bit", "gqa_reuse", "hot_ring"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be bool")
        if type(self.page_slots) is not int or self.page_slots <= 0:
            raise ValueError("page_slots must be positive")
        if type(self.cold_group) is not int or self.cold_group not in (1, 4, 8, 16):
            raise ValueError("cold_group must be 1, 4, 8 or 16")

    def to_dict(self):
        return asdict(self)


def resolve_config(value=None):
    if value is None:
        return OptimizationConfig()
    return value if isinstance(value, OptimizationConfig) else OptimizationConfig(**value)


def stage_configs():
    """Cumulative stages; individual switches can also be disabled independently."""
    stages = {"baseline": OptimizationConfig()}
    for name, changes in (
        ("p1", {"fused_quantize": True}),
        ("p2a", {"reuse_workspace": True}),
        ("p2b", {"paged_cold": True}),
        ("p3", {"cold_group": 8}),
        ("p4", {"specialized_4bit": True}),
        ("p4_gqa", {"gqa_reuse": True}),
    ):
        stages[name] = replace(next(reversed(stages.values())), **changes)
    # GQA remains a separate experiment, not a dependency of the ring buffer.
    stages["p5"] = replace(stages["p4"], hot_ring=True)
    return stages
