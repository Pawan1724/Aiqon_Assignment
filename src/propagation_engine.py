"""
propagation_engine.py  –  Stage-by-stage delay propagation model.

Design
------
The dataset does NOT contain intermediate stage timestamps.  Therefore
the propagation engine uses a **rule-based simulation** that:

1. Decomposes the total scheduled journey duration into configurable
   stages (departure → transit → customs/processing → last-mile delivery).
2. Allocates each stage a fraction of total scheduled duration.
3. Models additional delays per stage using weather multipliers,
   carrier reliability, congestion proxies, and random perturbation.
4. Propagates delays: each stage's start time is the previous stage's end.
   Incoming delay is NOT absorbed unless an explicit buffer exists.

This module clearly labels all outputs as **simulated / rule-based**.
No simulated results are mixed with observed data in model evaluation.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src import CONFIG, logger


# ------------------------------------------------------------------ #
#  Data classes                                                       #
# ------------------------------------------------------------------ #

@dataclass
class StageResult:
    """Result for a single shipment stage."""
    stage_name: str
    scheduled_start_min: float       # minutes from journey start
    scheduled_end_min: float
    scheduled_duration_min: float
    projected_start_min: float       # with incoming delay
    projected_end_min: float
    additional_delay_min: float      # delay generated in THIS stage
    cumulative_delay_min: float      # total delay arriving at end of stage
    factors: dict[str, Any] = field(default_factory=dict)


@dataclass
class PropagationResult:
    """Full propagation output for one shipment scenario."""
    stages: list[StageResult]
    total_scheduled_min: float
    total_projected_min: float
    total_delay_min: float
    is_simulated: bool = True        # always True for this engine
    assumptions: str = ""


# ------------------------------------------------------------------ #
#  Engine                                                              #
# ------------------------------------------------------------------ #

class DelayPropagationEngine:
    """
    Rule-based, inspectable stage-by-stage propagation engine.

    Parameters
    ----------
    stage_config : list[dict] | None
        Override default stages from settings.yaml.
    weather_multipliers : dict[str, float] | None
        Override default weather multipliers.
    """

    def __init__(
        self,
        stage_config: list[dict] | None = None,
        weather_multipliers: dict[str, float] | None = None,
    ):
        self.stages = stage_config or CONFIG["propagation"]["stages"]
        self.weather_mults = weather_multipliers or CONFIG["propagation"]["weather_multipliers"]

    # ----------------------------------------------------------------
    def propagate(
        self,
        scheduled_duration_min: float,
        departure_delay_min: float = 0.0,
        weather_departure: str = "clear",
        weather_arrival: str = "clear",
        carrier_on_time_rate: float = 0.90,
        route_type: str = "highway",
        custom_stage_delays: dict[str, float] | None = None,
        seed: int | None = None,
    ) -> PropagationResult:
        """
        Run the propagation simulation.

        Parameters
        ----------
        scheduled_duration_min : total scheduled journey time.
        departure_delay_min : observed or hypothetical departure delay.
        weather_departure : weather at origin.
        weather_arrival : weather at destination.
        carrier_on_time_rate : historical carrier reliability (0-1).
        route_type : highway / sea / air / rail.
        custom_stage_delays : {stage_name: extra_delay_minutes} for scenario sim.
        seed : random seed for any stochastic component.

        Returns
        -------
        PropagationResult
        """
        rng = np.random.default_rng(seed or CONFIG["project"]["random_seed"])

        stage_results: list[StageResult] = []
        cumulative_delay = departure_delay_min
        current_time = 0.0  # minutes from scheduled start

        for i, stage_cfg in enumerate(self.stages):
            name = stage_cfg["name"]
            frac = stage_cfg["default_fraction"]
            sched_dur = scheduled_duration_min * frac

            sched_start = current_time
            sched_end = current_time + sched_dur

            projected_start = sched_start + cumulative_delay

            # ----- compute additional delay for this stage -----
            add_delay = 0.0

            # Weather effect
            if name == "departure" or name == "transit":
                w = weather_departure
            else:
                w = weather_arrival if weather_arrival and weather_arrival != "nan" else "clear"
            weather_mult = self.weather_mults.get(w, 1.0)
            weather_add = sched_dur * (weather_mult - 1.0)
            add_delay += weather_add

            # Carrier reliability effect (lower rate → more delay)
            reliability_penalty = sched_dur * max(0, (1.0 - carrier_on_time_rate)) * 0.5
            add_delay += reliability_penalty

            # Route-type adjustment
            route_mult = {"sea": 1.08, "air": 0.95, "rail": 1.02, "highway": 1.0}.get(route_type, 1.0)
            route_add = sched_dur * (route_mult - 1.0)
            add_delay += max(0, route_add)

            # Custom scenario injection
            if custom_stage_delays and name in custom_stage_delays:
                add_delay += custom_stage_delays[name]

            # Small stochastic noise (± 2% of stage duration)
            noise = rng.normal(0, sched_dur * 0.02) if sched_dur > 0 else 0
            add_delay = max(0, add_delay + noise)

            cumulative_delay += add_delay
            projected_end = sched_end + cumulative_delay

            stage_results.append(StageResult(
                stage_name=name,
                scheduled_start_min=round(sched_start, 2),
                scheduled_end_min=round(sched_end, 2),
                scheduled_duration_min=round(sched_dur, 2),
                projected_start_min=round(projected_start, 2),
                projected_end_min=round(projected_end, 2),
                additional_delay_min=round(add_delay, 2),
                cumulative_delay_min=round(cumulative_delay, 2),
                factors={
                    "weather": w,
                    "weather_delay_min": round(weather_add, 2),
                    "carrier_penalty_min": round(reliability_penalty, 2),
                    "route_type_adj_min": round(max(0, route_add), 2),
                    "custom_inject_min": round(
                        custom_stage_delays.get(name, 0) if custom_stage_delays else 0, 2
                    ),
                },
            ))

            current_time = sched_end  # advance schedule clock

        total_projected = scheduled_duration_min + cumulative_delay
        assumptions = (
            "Stage fractions, weather multipliers, and carrier penalties "
            "are rule-based estimates. No intermediate timestamps exist in "
            "the dataset. All stage-level timings are SIMULATED."
        )

        return PropagationResult(
            stages=stage_results,
            total_scheduled_min=round(scheduled_duration_min, 2),
            total_projected_min=round(total_projected, 2),
            total_delay_min=round(cumulative_delay, 2),
            is_simulated=True,
            assumptions=assumptions,
        )

    # ----------------------------------------------------------------
    def simulate_what_if(
        self,
        base_result: PropagationResult,
        inject_stage: str,
        inject_delay_min: float,
    ) -> PropagationResult:
        """
        Re-run propagation with an additional delay injected at a specific stage.

        Parameters
        ----------
        base_result : the original propagation result (used only for params).
        inject_stage : stage name to inject delay into.
        inject_delay_min : minutes of additional delay.

        Returns a new PropagationResult.
        """
        # Reconstruct from base – extract original scheduled duration
        # We re-run propagate with the injection
        # For simplicity, re-use default parameters + injection
        logger.info(
            "What-if: injecting %.1f min delay at '%s'",
            inject_delay_min, inject_stage,
        )
        # This is a simplified re-propagation; the caller should supply
        # full parameters.  See prediction_service for the integrated call.
        custom = {inject_stage: inject_delay_min}
        return self.propagate(
            scheduled_duration_min=base_result.total_scheduled_min,
            custom_stage_delays=custom,
        )

    # ----------------------------------------------------------------
    def to_dataframe(self, result: PropagationResult) -> pd.DataFrame:
        """Convert a PropagationResult to a readable DataFrame."""
        records = []
        for s in result.stages:
            rec = {
                "stage": s.stage_name,
                "sched_start_min": s.scheduled_start_min,
                "sched_end_min": s.scheduled_end_min,
                "sched_duration_min": s.scheduled_duration_min,
                "proj_start_min": s.projected_start_min,
                "proj_end_min": s.projected_end_min,
                "stage_delay_min": s.additional_delay_min,
                "cumul_delay_min": s.cumulative_delay_min,
            }
            rec.update(s.factors)
            records.append(rec)
        df = pd.DataFrame(records)
        return df
