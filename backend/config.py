from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    db_path: str = "data/visionpilot.db"
    max_total_budget: float = 10_000.0
    meta_adapter_enabled: bool = False
    sim_seed: int = 0
    # LLM. Empty provider means the offline rule-based planner is used.
    llm_provider: str = ""
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key: str = field(default="", repr=False)  # never printed or logged
    agent_max_steps: int = 6
    uplift_customers: int = 20_000
    exploration_floor: float = 0.01  # minimum budget share for every ad (see docs/benchmark.md)

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        return cls(
            db_path=env("DB_PATH", cls.db_path),
            max_total_budget=float(env("MAX_TOTAL_BUDGET", cls.max_total_budget)),
            meta_adapter_enabled=env("META_ADAPTER_ENABLED", "false").strip().lower() in {"1", "true", "yes"},
            sim_seed=int(env("SIM_SEED", cls.sim_seed)),
            llm_provider=env("LLM_PROVIDER", "").strip().lower(),
            llm_model=env("LLM_MODEL", "").strip(),
            llm_base_url=env("LLM_BASE_URL", "").strip(),
            llm_api_key=env("LLM_API_KEY", "").strip(),
            agent_max_steps=int(env("AGENT_MAX_STEPS", cls.agent_max_steps)),
            uplift_customers=int(env("UPLIFT_CUSTOMERS", cls.uplift_customers)),
            exploration_floor=float(env("EXPLORATION_FLOOR", cls.exploration_floor)),
        )
