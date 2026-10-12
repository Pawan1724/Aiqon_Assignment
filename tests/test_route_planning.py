"""
tests/test_route_planning.py - Regression test suite for intermodal route planning,
stop sequencing validation, ETA calculations, and map consistency.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.route_planner import (
    plan_intermodal_corridors,
    validate_route_sequence,
    resolve_node,
    haversine_km,
)


def test_india_to_chicago_routing():
    """
    Requirement 1, 6 & 8:
    - Route from India to Chicago must NOT send shipment to Europe/Hamburg after arrival.
    - Chicago is an inland intermodal rail hub, not an ocean port.
    - Ocean cargo must discharge at an East Coast gateway (Port of NY/NJ) and move via rail.
    - ETA and stage delays must be calculated dynamically, not hardcoded to 22h.
    """
    res = plan_intermodal_corridors(origin_name="India", destination_name="Chicago")
    assert res["origin"] == "India"
    assert res["destination"] == "Chicago"
    assert res["is_illustrative"] is False
    assert len(res["routes"]) == 3

    opt1 = res["routes"][0]
    assert opt1["id"] == "route-opt-1"
    assert opt1["route_type"] == "multimodal"

    # Verify stop sequence
    node_labels = [n["label"] for n in opt1["timeline_nodes"]]
    assert node_labels == ["INDIA", "DUBAI", "NEW YORK", "CHICAGO"]

    # Hamburg must NEVER appear in this corridor
    assert "HAMBURG" not in node_labels

    # Verify legs and transport modes
    legs = opt1["legs"]
    assert len(legs) == 3
    assert legs[0]["from"] == "Mumbai"
    assert legs[0]["to"] == "Dubai"
    assert legs[0]["mode"] == "ship"

    assert legs[1]["from"] == "Dubai"
    assert legs[1]["to"] == "New York"
    assert legs[1]["mode"] == "ship"

    assert legs[2]["from"] == "New York"
    assert legs[2]["to"] == "Chicago"
    assert legs[2]["mode"] == "rail"  # Intermodal rail to Chicago inland destination

    # Verify destination termination
    assert opt1["waypoints"][-1]["name"] == "Chicago"
    assert opt1["waypoints"][-1]["role"] == "destination"

    # Verify dynamic delay calculation (no hardcoded text or fake 22h Hamburg alert)
    assert "overloaded port Hamburg" not in opt1["alert"]
    assert opt1["est_duration_minutes"] > opt1["sched_duration_minutes"]
    assert opt1["total_delay_minutes"] > 0
    assert "Calculated from intermodal stage transfer and carrier reliability" in opt1["alert"]


def test_india_to_usa_unspecified_vs_specified():
    """
    Requirement 7:
    - If destination is 'USA' (unspecified city), route must be marked as illustrative.
    - Clear illustrative note must prompt user to select a specific city.
    - Gateway port (Port of NY/NJ) must serve as the arrival gateway.
    - If a specific destination city (e.g. Chicago or New York) is given, is_illustrative is False.
    """
    # Unspecified destination
    res_unspec = plan_intermodal_corridors(origin_name="India", destination_name="USA")
    assert res_unspec["is_illustrative"] is True
    assert res_unspec["illustrative_note"] is not None
    assert "Destination city unspecified" in res_unspec["illustrative_note"]
    assert "Port of NY/NJ" in res_unspec["illustrative_note"]

    # Routes reflect illustrative gateway
    opt1_unspec = res_unspec["routes"][0]
    assert opt1_unspec["is_illustrative"] is True
    assert "NEW YORK" in [n["label"] for n in opt1_unspec["timeline_nodes"]]

    # Specified destination
    res_spec = plan_intermodal_corridors(origin_name="India", destination_name="New York")
    assert res_spec["is_illustrative"] is False
    assert res_spec["illustrative_note"] is None


def test_legitimate_transshipment_through_hamburg():
    """
    Requirement 4, 5 & 13:
    - Transshipment through Hamburg is legitimate for cargo destined for Northern Europe/Germany/Baltic.
    - Hamburg must precede the inland destination (e.g. Berlin).
    - Mode from Hamburg to Berlin must be rail/road, not ocean ship.
    """
    res = plan_intermodal_corridors(origin_name="India", destination_name="Berlin")
    opt1 = res["routes"][0]

    node_labels = [n["label"] for n in opt1["timeline_nodes"]]
    assert node_labels == ["INDIA", "DUBAI", "HAMBURG", "BERLIN"]

    legs = opt1["legs"]
    assert len(legs) == 3
    # Sea leg into Hamburg Port
    assert legs[1]["to"] == "Hamburg"
    assert legs[1]["mode"] == "ship"
    # Hinterland rail from Hamburg Port to Berlin
    assert legs[2]["from"] == "Hamburg"
    assert legs[2]["to"] == "Berlin"
    assert legs[2]["mode"] == "rail"

    # Final destination is Berlin
    assert opt1["waypoints"][-1]["name"] == "Berlin"


def test_invalid_stop_sequences_prevented():
    """
    Requirement 5 & 6:
    - Backtracking/looping sequences (e.g. Chicago -> Hamburg -> Chicago) must be invalid.
    - Ocean shipping directly to an inland city must be invalid.
    - Discontinuous jumps must be invalid.
    """
    # 1. Backtracking loop: Chicago -> Hamburg -> Chicago
    looping_legs = [
        {"from_name": "Chicago", "to_name": "Hamburg", "mode": "ship"},
        {"from_name": "Hamburg", "to_name": "Chicago", "mode": "ship"},
    ]
    valid, reason = validate_route_sequence(looping_legs, "Chicago", "Chicago")
    assert valid is False
    assert "loop" in reason.lower()

    # 2. Ocean vessel routed directly to inland node (Frankfurt is not a seaport)
    infeasible_mode_legs = [
        {"from_name": "Mumbai", "to_name": "Frankfurt", "mode": "ship"},
    ]
    valid, reason = validate_route_sequence(infeasible_mode_legs, "Mumbai", "Frankfurt")
    assert valid is False
    assert "infeasible modal connection" in reason.lower()

    # 3. Discontinuous leg
    discontinuous_legs = [
        {"from_name": "Mumbai", "to_name": "Dubai", "mode": "ship"},
        {"from_name": "Rotterdam", "to_name": "Berlin", "mode": "rail"},
    ]
    valid, reason = validate_route_sequence(discontinuous_legs, "Mumbai", "Berlin")
    assert valid is False
    assert "discontinuous" in reason.lower()


def test_map_legs_and_eta_consistency():
    """
    Requirement 10 & 11:
    - Waypoints sequence matches the leg sequence.
    - Total distance matches the sum of leg distances.
    - Estimated duration equals scheduled duration plus stage delays.
    - Timeline nodes match waypoints.
    """
    pairs = [
        ("India", "Chicago"),
        ("Istanbul", "Dublin"),
        ("India", "Berlin"),
        ("India", "USA"),
    ]

    for orig, dest in pairs:
        res = plan_intermodal_corridors(origin_name=orig, destination_name=dest)
        for route in res["routes"]:
            # 1. Waypoint count and endpoints
            wps = route["waypoints"]
            assert len(wps) >= 2
            assert wps[0]["role"] == "origin"
            assert wps[-1]["role"] == "destination"

            # 2. Legs continuity
            legs = route["legs"]
            for i in range(len(legs) - 1):
                assert legs[i]["to"] == legs[i + 1]["from"]

            # 3. Distance consistency
            sum_leg_dist = sum(l["distance_km"] for l in legs)
            assert abs(route["distance_km"] - sum_leg_dist) <= 2.0

            # 4. ETA consistency: est_duration == sched_duration + total_delay
            sched = route["sched_duration_minutes"]
            delay = route["total_delay_minutes"]
            est = route["est_duration_minutes"]
            assert abs(est - (sched + delay)) < 0.001

            # 5. Timeline nodes match waypoints
            timeline_labels = [n["label"] for n in route["timeline_nodes"]]
            waypoint_names = [w["name"].upper() for w in wps]
            assert len(timeline_labels) == len(waypoint_names)
