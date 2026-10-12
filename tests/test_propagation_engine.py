"""Tests for the stage-by-stage delay propagation engine."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.propagation_engine import DelayPropagationEngine


@pytest.fixture
def engine():
    return DelayPropagationEngine(
        stage_config=[
            {"name": "departure", "default_fraction": 0.1},
            {"name": "transit", "default_fraction": 0.8},
            {"name": "delivery", "default_fraction": 0.1},
        ],
        weather_multipliers={"clear": 1.0, "rain": 1.1}
    )


def test_propagate_basic(engine):
    """Test basic propagation with no external delays."""
    res = engine.propagate(
        scheduled_duration_min=100.0,
        departure_delay_min=0.0,
        weather_departure="clear",
        weather_arrival="clear",
        carrier_on_time_rate=1.0,
        route_type="highway",
        seed=42
    )

    assert res.total_scheduled_min == 100.0
    assert len(res.stages) == 3
    assert res.stages[0].scheduled_duration_min == 10.0
    assert res.stages[1].scheduled_duration_min == 80.0
    assert res.stages[2].scheduled_duration_min == 10.0
    assert res.is_simulated is True


def test_propagate_with_departure_delay(engine):
    """Test propagation with an initial departure delay."""
    res = engine.propagate(
        scheduled_duration_min=100.0,
        departure_delay_min=30.0,
        weather_departure="clear",
        weather_arrival="clear",
        carrier_on_time_rate=1.0,
        route_type="highway",
        seed=42
    )
    
    # Delay should cascade through all stages
    assert res.stages[0].projected_start_min == 30.0
    # Final delay should be >= 30
    assert res.total_delay_min >= 30.0


def test_propagate_with_weather_delay(engine):
    """Test that bad weather adds delay."""
    res_clear = engine.propagate(
        scheduled_duration_min=100.0,
        weather_departure="clear",
        carrier_on_time_rate=1.0,
        seed=42
    )
    res_rain = engine.propagate(
        scheduled_duration_min=100.0,
        weather_departure="rain",
        carrier_on_time_rate=1.0,
        seed=42
    )
    
    assert res_rain.total_delay_min > res_clear.total_delay_min


def test_simulate_what_if(engine):
    """Test injecting delay into a specific stage."""
    base = engine.propagate(scheduled_duration_min=100.0, seed=42)
    injected = engine.simulate_what_if(base, "transit", 60.0)
    
    assert injected.total_delay_min > base.total_delay_min
    assert "transit" in [s.stage_name for s in injected.stages]
    # Check that transit stage has the extra delay
    for s in injected.stages:
        if s.stage_name == "transit":
            assert s.factors["custom_inject_min"] == 60.0
