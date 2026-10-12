"""
src/route_planner.py - Intermodal Logistics Routing, ETA Calculation, and Stage Delay Engine.

Provides:
- Node capabilities (seaports, airports, inland cities, rail terminals)
- Feasible multimodal corridor generation (Sea+Rail, Air+Truck, Air Express)
- Stop sequencing validation (prevents backtracking, invalid modal connections, loops)
- Transparent, leg-by-leg ETA and stage delay calculations (replaces hardcoded delays)
- Unspecified destination handling with illustrative routing labels
"""

import math
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two coordinates in kilometers."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return max(20.0, round(R * c, 1))


def format_duration_display(minutes: float) -> str:
    """Format minutes to human-readable logistics format (e.g. '21d 6h', '18 hours', '45 mins')."""
    total_mins = max(15, int(round(minutes)))
    total_hours = total_mins // 60
    rem_mins = total_mins % 60
    days = total_hours // 24
    hours = total_hours % 24

    if days > 0:
        if hours > 0:
            return f"{days}d {hours}h"
        return f"{days} days" if days > 1 else "1 day"
    elif hours > 0:
        if rem_mins >= 20:
            return f"{hours}h {rem_mins}m"
        return f"{hours} hours" if hours > 1 else "1 hour"
    else:
        return f"{rem_mins} mins"


def format_inr(amount_inr: float) -> str:
    """Format currency using Indian numbering standard (Lakhs and Crores, e.g. ₹5,25,000)."""
    val = int(round(amount_inr))
    s = str(val)
    if len(s) <= 3:
        return f"₹{s}"
    last3 = s[-3:]
    remaining = s[:-3]
    groups = []
    while len(remaining) > 2:
        groups.append(remaining[-2:])
        remaining = remaining[:-2]
    if remaining:
        groups.append(remaining)
    groups.reverse()
    return f"₹{','.join(groups)},{last3}"


# =========================================================================
# Logistics Infrastructure Database
# =========================================================================

LOGISTICS_NODES: Dict[str, Dict[str, Any]] = {
    # India
    "mumbai": {
        "display_name": "Mumbai",
        "country": "India",
        "coords": [18.9553, 72.8258],
        "is_seaport": True,
        "port_title": "Jawaharlal Nehru Port (Nhava Sheva)",
        "is_airport": True,
        "airport_code": "BOM",
        "nearest_seaport": "mumbai",
        "nearest_airport": "mumbai",
    },
    "delhi": {
        "display_name": "Delhi",
        "country": "India",
        "coords": [28.6139, 77.2090],
        "is_seaport": False,
        "port_title": None,
        "is_airport": True,
        "airport_code": "DEL",
        "nearest_seaport": "mumbai",
        "nearest_airport": "delhi",
    },
    "india": {
        "display_name": "India",
        "country": "India",
        "coords": [18.9553, 72.8258],
        "is_seaport": True,
        "port_title": "Nhava Sheva Gateway (India)",
        "is_airport": True,
        "airport_code": "BOM",
        "nearest_seaport": "mumbai",
        "nearest_airport": "mumbai",
    },

    # Middle East / Transshipment
    "dubai": {
        "display_name": "Dubai",
        "country": "UAE",
        "coords": [25.0112, 55.0617],
        "is_seaport": True,
        "port_title": "Jebel Ali Port (Transshipment Hub)",
        "is_airport": True,
        "airport_code": "DXB",
        "nearest_seaport": "dubai",
        "nearest_airport": "dubai",
    },
    "cairo": {
        "display_name": "Cairo",
        "country": "Egypt",
        "coords": [31.2001, 32.3167],
        "is_seaport": True,
        "port_title": "Port Said (Suez Canal Transit)",
        "is_airport": True,
        "airport_code": "CAI",
        "nearest_seaport": "cairo",
        "nearest_airport": "cairo",
    },

    # Europe
    "rotterdam": {
        "display_name": "Rotterdam",
        "country": "Netherlands",
        "coords": [51.9540, 4.1350],
        "is_seaport": True,
        "port_title": "Port of Rotterdam (Mainport Europe)",
        "is_airport": False,
        "nearest_seaport": "rotterdam",
        "nearest_airport": "amsterdam",
    },
    "hamburg": {
        "display_name": "Hamburg",
        "country": "Germany",
        "coords": [53.5459, 9.9669],
        "is_seaport": True,
        "port_title": "Hamburg Container Terminal (Burchardkai)",
        "is_airport": True,
        "airport_code": "HAM",
        "nearest_seaport": "hamburg",
        "nearest_airport": "hamburg",
    },
    "frankfurt": {
        "display_name": "Frankfurt",
        "country": "Germany",
        "coords": [50.0379, 8.5622],
        "is_seaport": False,
        "port_title": None,
        "is_airport": True,
        "airport_code": "FRA",
        "nearest_seaport": "rotterdam",
        "nearest_airport": "frankfurt",
    },
    "paris": {
        "display_name": "Paris",
        "country": "France",
        "coords": [48.8566, 2.3522],
        "is_seaport": False,
        "port_title": None,
        "is_airport": True,
        "airport_code": "CDG",
        "nearest_seaport": "caen",
        "nearest_airport": "paris",
    },
    "caen": {
        "display_name": "Caen",
        "country": "France",
        "coords": [49.2800, -0.2500],
        "is_seaport": True,
        "port_title": "Port of Ouistreham/Caen",
        "is_airport": False,
        "nearest_seaport": "caen",
        "nearest_airport": "nantes",
    },
    "nantes": {
        "display_name": "Nantes",
        "country": "France",
        "coords": [47.2184, -1.5536],
        "is_seaport": False,
        "port_title": None,
        "is_airport": True,
        "airport_code": "NTE",
        "nearest_seaport": "caen",
        "nearest_airport": "nantes",
    },
    "london": {
        "display_name": "London",
        "country": "UK",
        "coords": [51.5074, 0.4543],
        "is_seaport": True,
        "port_title": "London Gateway Port",
        "is_airport": True,
        "airport_code": "LHR",
        "nearest_seaport": "london",
        "nearest_airport": "london",
    },
    "dublin": {
        "display_name": "Dublin",
        "country": "Ireland",
        "coords": [53.3463, -6.2208],
        "is_seaport": True,
        "port_title": "Dublin Port",
        "is_airport": True,
        "airport_code": "DUB",
        "nearest_seaport": "dublin",
        "nearest_airport": "dublin",
    },
    "istanbul": {
        "display_name": "Istanbul",
        "country": "Turkey",
        "coords": [40.9700, 28.7180],
        "is_seaport": True,
        "port_title": "Port of Ambarli",
        "is_airport": True,
        "airport_code": "IST",
        "nearest_seaport": "istanbul",
        "nearest_airport": "istanbul",
    },
    "berlin": {
        "display_name": "Berlin",
        "country": "Germany",
        "coords": [52.5200, 13.4050],
        "is_seaport": False,
        "port_title": None,
        "is_airport": True,
        "airport_code": "BER",
        "nearest_seaport": "hamburg",
        "nearest_airport": "berlin",
    },
    "germany": {
        "display_name": "Germany",
        "country": "Germany",
        "coords": [53.5459, 9.9669],
        "is_seaport": True,
        "port_title": "Port of Hamburg (German Gateway)",
        "is_airport": True,
        "airport_code": "HAM",
        "nearest_seaport": "hamburg",
        "nearest_airport": "frankfurt",
    },

    # North America / USA
    "new york": {
        "display_name": "New York",
        "country": "USA",
        "coords": [40.6895, -74.1745],
        "is_seaport": True,
        "port_title": "Port of New York & New Jersey (East Coast Gateway)",
        "is_airport": True,
        "airport_code": "JFK",
        "nearest_seaport": "new york",
        "nearest_airport": "new york",
    },
    "chicago": {
        "display_name": "Chicago",
        "country": "USA",
        "coords": [41.8781, -87.6298],
        "is_seaport": False,
        "port_title": "Chicago Intermodal Rail Terminal",
        "is_rail_hub": True,
        "is_airport": True,
        "airport_code": "ORD",
        "nearest_seaport": "new york",
        "nearest_airport": "chicago",
    },
    "los angeles": {
        "display_name": "Los Angeles",
        "country": "USA",
        "coords": [33.7432, -118.2673],
        "is_seaport": True,
        "port_title": "Port of Los Angeles / Long Beach (West Coast Gateway)",
        "is_airport": True,
        "airport_code": "LAX",
        "nearest_seaport": "los angeles",
        "nearest_airport": "los angeles",
    },
    "dallas": {
        "display_name": "Dallas",
        "country": "USA",
        "coords": [32.7767, -96.7970],
        "is_seaport": False,
        "port_title": None,
        "is_airport": True,
        "airport_code": "DFW",
        "nearest_seaport": "houston",
        "nearest_airport": "dallas",
    },
    "houston": {
        "display_name": "Houston",
        "country": "USA",
        "coords": [29.7604, -95.3698],
        "is_seaport": True,
        "port_title": "Port of Houston",
        "is_airport": True,
        "airport_code": "IAH",
        "nearest_seaport": "houston",
        "nearest_airport": "houston",
    },
    "usa": {
        "display_name": "USA (Gateway: Port of NY/NJ)",
        "country": "USA",
        "coords": [40.6895, -74.1745],
        "is_seaport": True,
        "port_title": "Port of NY/NJ (US Gateway)",
        "is_airport": True,
        "airport_code": "JFK",
        "nearest_seaport": "new york",
        "nearest_airport": "new york",
        "is_unspecified": True,
    },
    "united states": {
        "display_name": "USA (Gateway: Port of NY/NJ)",
        "country": "USA",
        "coords": [40.6895, -74.1745],
        "is_seaport": True,
        "port_title": "Port of NY/NJ (US Gateway)",
        "is_airport": True,
        "airport_code": "JFK",
        "nearest_seaport": "new york",
        "nearest_airport": "new york",
        "is_unspecified": True,
    },

    # Asia
    "singapore": {
        "display_name": "Singapore",
        "country": "Singapore",
        "coords": [1.2640, 103.8400],
        "is_seaport": True,
        "port_title": "Port of Singapore (PSA)",
        "is_airport": True,
        "airport_code": "SIN",
        "nearest_seaport": "singapore",
        "nearest_airport": "singapore",
    },
    "hong kong": {
        "display_name": "Hong Kong",
        "country": "Hong Kong",
        "coords": [22.3193, 114.1694],
        "is_seaport": True,
        "port_title": "Port of Hong Kong",
        "is_airport": True,
        "airport_code": "HKG",
        "nearest_seaport": "hong kong",
        "nearest_airport": "hong kong",
    },
    "tokyo": {
        "display_name": "Tokyo",
        "country": "Japan",
        "coords": [35.6190, 139.7790],
        "is_seaport": True,
        "port_title": "Port of Tokyo",
        "is_airport": True,
        "airport_code": "NRT",
        "nearest_seaport": "tokyo",
        "nearest_airport": "tokyo",
    },
    "sydney": {
        "display_name": "Sydney",
        "country": "Australia",
        "coords": [-33.9500, 151.2100],
        "is_seaport": True,
        "port_title": "Port Botany (Sydney)",
        "is_airport": True,
        "airport_code": "SYD",
        "nearest_seaport": "sydney",
        "nearest_airport": "sydney",
    },
}


def resolve_node(name: str) -> Dict[str, Any]:
    """Resolve location string to a standardized logistics node dictionary."""
    clean = name.strip().lower()
    if clean in LOGISTICS_NODES:
        return LOGISTICS_NODES[clean]
    for k, v in LOGISTICS_NODES.items():
        if k in clean or clean in k:
            return v
    # Generic fallback
    return {
        "display_name": name.strip().title(),
        "country": "Unknown",
        "coords": [48.5, 10.0],
        "is_seaport": False,
        "port_title": f"{name.title()} Terminal",
        "is_airport": True,
        "airport_code": name[:3].upper(),
        "nearest_seaport": "rotterdam",
        "nearest_airport": clean,
    }


# =========================================================================
# Route Sequencing & Validation
# =========================================================================

def validate_route_sequence(legs: List[Dict[str, Any]], expected_origin: str, expected_destination: str) -> Tuple[bool, str]:
    """
    Validate that a proposed stop sequence is operationally feasible:
    - Must start at expected_origin and end at expected_destination.
    - Each leg's from_node must match the previous leg's to_node (continuity).
    - Sea mode is forbidden directly to inland cities without port transfer.
    - No loops or backtracking (e.g. Chicago -> Hamburg -> Chicago is rejected).
    """
    if not legs:
        return False, "Route contains no legs."

    # Origin & Destination termination
    first_node = legs[0]["from_name"].strip().lower()
    last_node = legs[-1]["to_name"].strip().lower()
    exp_orig = expected_origin.strip().lower()
    exp_dest = expected_destination.strip().lower()

    if exp_orig not in first_node and first_node not in exp_orig:
        return False, f"Route origin '{first_node}' does not match expected '{expected_origin}'."

    # Check destination match (allow 'usa' matching 'new york' if illustrative gateway)
    dest_matches = (exp_dest in last_node or last_node in exp_dest)
    if not dest_matches and exp_dest in ["usa", "united states"]:
        dest_matches = True
    if not dest_matches:
        return False, f"Route terminates at '{last_node}', but expected destination is '{expected_destination}'."

    visited_nodes = [legs[0]["from_name"].lower()]
    for i, leg in enumerate(legs):
        # Continuity
        if i > 0 and leg["from_name"].lower() != legs[i - 1]["to_name"].lower():
            return False, f"Discontinuous leg at index {i}: {legs[i-1]['to_name']} -> {leg['from_name']}."

        to_n = leg["to_name"].lower()
        # Backtracking check: no node revisited
        if to_n in visited_nodes:
            return False, f"Invalid loop detected: '{to_n}' is revisited in the route sequence."
        visited_nodes.append(to_n)

        # Modal feasibility: sea freight cannot connect directly to inland destinations
        if leg["mode"] == "ship":
            to_node_info = resolve_node(leg["to_name"])
            if not to_node_info.get("is_seaport", False):
                return False, f"Infeasible modal connection: Ocean vessel routed directly to inland node '{leg['to_name']}'."

    return True, "Sequence is valid and operationally feasible."


# =========================================================================
# Multimodal Corridor Generation
# =========================================================================

def plan_intermodal_corridors(
    origin_name: str,
    destination_name: str,
    cbm: float = 24.0,
    wm: float = 12.0,
    optimise_for: str = "price",
    weather: str = "rain",
    priority: str = "standard",
    carrier_rate: float = 0.88,
) -> Dict[str, Any]:
    """
    Generate 3 distinct, operationally feasible transport corridors:
    - Option 1: Multimodal Ocean & Inland Rail/Road (Standard Sea Freight)
    - Option 2: Air Freight & Regional Transfer (Consolidated Air Cargo)
    - Option 3: Direct Air Express (Fast Courier Flight)

    Calculates exact leg-by-leg durations, dwell times, and stage delays.
    """
    orig = resolve_node(origin_name)
    dest = resolve_node(destination_name)
    orig_clean = origin_name.strip().lower()
    dest_clean = destination_name.strip().lower()

    is_unspecified = dest.get("is_unspecified", False) or dest_clean in ["usa", "united states"]
    illustrative_note = None
    if is_unspecified:
        illustrative_note = (
            "Destination city unspecified. Showing illustrative routing to US East Coast Gateway (Port of NY/NJ). "
            "Please select a specific destination city (e.g., Chicago, New York, Los Angeles) for exact door-to-door transit time."
        )

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Direct geographic baseline
    direct_dist_km = haversine_km(orig["coords"][0], orig["coords"][1], dest["coords"][0], dest["coords"][1])

    # =========================================================================
    # OPTION 1: Multimodal Ocean & Inland Rail/Road
    # =========================================================================
    opt1_legs = _build_multimodal_ocean_legs(orig, dest, orig_clean, dest_clean, direct_dist_km)
    opt1 = _evaluate_route_option(
        route_id="route-opt-1",
        title="Multimodal Ocean & Inland Rail",
        route_type="multimodal",
        is_recommended=True,
        is_selected=True,
        legs=opt1_legs,
        origin_node=orig,
        dest_node=dest,
        cbm=cbm,
        wm=wm,
        cost_per_km_rate=1.45,
        base_handling_fee=450.0,
        weather=weather,
        carrier_rate=carrier_rate,
        is_illustrative=is_unspecified,
        illustrative_note=illustrative_note,
    )

    # =========================================================================
    # OPTION 2: Air Freight & Regional Transfer
    # =========================================================================
    opt2_legs = _build_air_freight_legs(orig, dest, direct_dist_km)
    opt2 = _evaluate_route_option(
        route_id="route-opt-2",
        title="Air Freight & Regional Transfer",
        route_type="air_sea",
        is_recommended=False,
        is_selected=False,
        legs=opt2_legs,
        origin_node=orig,
        dest_node=dest,
        cbm=cbm,
        wm=wm,
        cost_per_km_rate=2.60,
        base_handling_fee=850.0,
        weather=weather,
        carrier_rate=min(0.95, carrier_rate + 0.05),
        is_illustrative=is_unspecified,
        illustrative_note=illustrative_note,
    )

    # =========================================================================
    # OPTION 3: Direct Air Express
    # =========================================================================
    opt3_legs = _build_direct_air_legs(orig, dest, direct_dist_km)
    opt3 = _evaluate_route_option(
        route_id="route-opt-3",
        title="Direct Air Express",
        route_type="air_direct",
        is_recommended=False,
        is_selected=False,
        legs=opt3_legs,
        origin_node=orig,
        dest_node=dest,
        cbm=cbm,
        wm=wm,
        cost_per_km_rate=3.80,
        base_handling_fee=1200.0,
        weather=weather,
        carrier_rate=0.96,
        badge="Next day delivery guarantee" if opt3_legs[0]["dist_km"] < 6000 else "Express Air Priority",
        is_illustrative=is_unspecified,
        illustrative_note=illustrative_note,
    )

    options = [opt1, opt2, opt3]
    if optimise_for == "time":
        options.sort(key=lambda x: x["est_duration_minutes"])
    elif optimise_for == "reliability":
        options.sort(key=lambda x: x["reliability_num"], reverse=True)

    return {
        "origin": orig["display_name"],
        "destination": dest["display_name"],
        "is_illustrative": is_unspecified,
        "illustrative_note": illustrative_note,
        "routes": options,
    }


# =========================================================================
# Leg Construction Helpers
# =========================================================================

def _build_multimodal_ocean_legs(
    orig: Dict[str, Any],
    dest: Dict[str, Any],
    orig_clean: str,
    dest_clean: str,
    direct_dist_km: float,
) -> List[Dict[str, Any]]:
    """Build operationally realistic sequence of multimodal legs."""
    legs: List[Dict[str, Any]] = []

    # 1. Exact preset for Istanbul to Dublin demo
    if "istanbul" in orig_clean and "dublin" in dest_clean:
        paris = LOGISTICS_NODES["paris"]
        caen = LOGISTICS_NODES["caen"]
        legs.append(_create_leg(orig, paris, "truck", "Continental Road Freight (Highway)", 180.0))
        legs.append(_create_leg(paris, caen, "truck", "Normandy Drayage Transfer", 90.0))
        legs.append(_create_leg(caen, dest, "ship", "Celtic Sea Ro-Ro Ferry Transit", 420.0))
        return legs

    # 2. Transpacific / Transatlantic from Asia/India to North America (Chicago / USA)
    if "india" in orig_clean or "mumbai" in orig_clean:
        if "chicago" in dest_clean:
            # India -> Dubai (Port) -> NY/NJ (Port) -> Chicago (Rail)
            mumbai = LOGISTICS_NODES["mumbai"]
            dubai = LOGISTICS_NODES["dubai"]
            nynj = LOGISTICS_NODES["new york"]
            chicago = LOGISTICS_NODES["chicago"]

            legs.append(_create_leg(mumbai, dubai, "ship", "Arabian Sea Feeder Voyage", 1440.0))
            legs.append(_create_leg(dubai, nynj, "ship", "Suez & North Atlantic Ocean Crossing", 2160.0))
            legs.append(_create_leg(nynj, chicago, "rail", "Intermodal Double-Stack Rail Corridors", 720.0))
            return legs

        if "usa" in dest_clean or "united states" in dest_clean or "new york" in dest_clean:
            # India -> Dubai (Port) -> NY/NJ Gateway (Port)
            mumbai = LOGISTICS_NODES["mumbai"]
            dubai = LOGISTICS_NODES["dubai"]
            nynj = LOGISTICS_NODES["new york"]

            legs.append(_create_leg(mumbai, dubai, "ship", "Arabian Sea Feeder Voyage", 1440.0))
            legs.append(_create_leg(dubai, nynj, "ship", "Suez & North Atlantic Ocean Crossing", 2160.0))
            return legs

        if "dublin" in dest_clean:
            # India -> Dubai (Port) -> Rotterdam/Caen (Port) -> Dublin (Port)
            mumbai = LOGISTICS_NODES["mumbai"]
            dubai = LOGISTICS_NODES["dubai"]
            caen = LOGISTICS_NODES["caen"]
            legs.append(_create_leg(mumbai, dubai, "ship", "Arabian Sea Feeder Voyage", 1440.0))
            legs.append(_create_leg(dubai, caen, "ship", "Suez & Mediterranean Maritime Transit", 1800.0))
            legs.append(_create_leg(caen, dest, "ship", "Celtic Sea Ro-Ro Ferry Transit", 420.0))
            return legs

    # 3. Legitimate Transshipment via Hamburg (e.g. to Germany / Scandinavia / Baltic)
    if "hamburg" in dest_clean or dest.get("country") == "Germany":
        dubai = LOGISTICS_NODES["dubai"]
        hamburg = LOGISTICS_NODES["hamburg"]
        legs.append(_create_leg(orig, dubai, "ship", "Export Seaborne Feeder", 1440.0))
        if dest_clean in ["hamburg", "germany"]:
            legs.append(_create_leg(dubai, dest, "ship", "North Sea Maritime Container Transit", 1920.0))
        else:
            legs.append(_create_leg(dubai, hamburg, "ship", "North Sea Maritime Container Transit", 1920.0))
            legs.append(_create_leg(hamburg, dest, "rail", "Hinterland Rail Shuttle", 360.0))
        return legs

    # 4. Standard Regional or Intercontinental fallback
    if direct_dist_km > 3500:
        dubai = LOGISTICS_NODES["dubai"]
        legs.append(_create_leg(orig, dubai, "ship", "Primary Maritime Export Leg", 1440.0))
        legs.append(_create_leg(dubai, dest, "ship", "Deep Sea Container Voyage", 1800.0))
    else:
        # Regional highway / coastal
        legs.append(_create_leg(orig, dest, "truck", "Regional Freight Corridor", 360.0))

    return legs


def _build_air_freight_legs(orig: Dict[str, Any], dest: Dict[str, Any], direct_dist_km: float) -> List[Dict[str, Any]]:
    """Build 2-leg Air Cargo option via international air hub."""
    legs: List[Dict[str, Any]] = []
    # Pick intermediate hub
    hub = LOGISTICS_NODES["dubai"] if direct_dist_km > 4000 else LOGISTICS_NODES["frankfurt"]
    legs.append(_create_leg(orig, hub, "airplane", "International Air Cargo Leg 1", 360.0))
    legs.append(_create_leg(hub, dest, "airplane", "International Air Cargo Leg 2", 360.0))
    return legs


def _build_direct_air_legs(orig: Dict[str, Any], dest: Dict[str, Any], direct_dist_km: float) -> List[Dict[str, Any]]:
    """Build single direct air express flight leg."""
    return [_create_leg(orig, dest, "airplane", "Direct Point-to-Point Express Flight", 120.0)]


def _create_leg(from_n: Dict[str, Any], to_n: Dict[str, Any], mode: str, desc: str, dwell_min: float) -> Dict[str, Any]:
    """Calculate leg distance and travel time based on mode physics."""
    dist_km = haversine_km(from_n["coords"][0], from_n["coords"][1], to_n["coords"][0], to_n["coords"][1])

    # Speed parameters
    if mode in ["airplane", "air"]:
        speed_kmh = 800.0
    elif mode in ["train", "rail"]:
        speed_kmh = 65.0
    elif mode in ["ship", "sea"]:
        speed_kmh = 35.0
    else:
        speed_kmh = 70.0  # truck / highway

    travel_min = (dist_km / speed_kmh) * 60.0
    sched_dur_min = travel_min + dwell_min

    return {
        "from_name": from_n["display_name"],
        "from_coords": from_n["coords"],
        "to_name": to_n["display_name"],
        "to_coords": to_n["coords"],
        "mode": mode,
        "description": desc,
        "dist_km": dist_km,
        "travel_min": travel_min,
        "dwell_min": dwell_min,
        "sched_dur_min": sched_dur_min,
        "to_node_info": to_n,
    }


# =========================================================================
# Route Option Evaluation & Delay Synthesis
# =========================================================================

def _evaluate_route_option(
    route_id: str,
    title: str,
    route_type: str,
    is_recommended: bool,
    is_selected: bool,
    legs: List[Dict[str, Any]],
    origin_node: Dict[str, Any],
    dest_node: Dict[str, Any],
    cbm: float,
    wm: float,
    cost_per_km_rate: float,
    base_handling_fee: float,
    weather: str,
    carrier_rate: float,
    badge: Optional[str] = None,
    is_illustrative: bool = False,
    illustrative_note: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute exact totals, timeline nodes, waypoints, and transparent stage delays."""
    total_dist_km = sum(l["dist_km"] for l in legs)
    total_sched_dur_min = sum(l["sched_dur_min"] for l in legs)

    # Calculate transparent stage delays
    stage_delays = []
    delay_markers = []

    weather_penalty_map = {"storm": 480.0, "rain": 180.0, "fog": 120.0, "clear": 0.0}
    weather_pen = weather_penalty_map.get(weather.lower(), 60.0)

    for i, leg in enumerate(legs):
        leg_delay_min = 0.0
        # Weather impact on long maritime or air legs
        if leg["mode"] in ["ship", "airplane"] and weather_pen > 0:
            stage_delays.append({
                "stage": f"Leg {i+1}: {leg['from_name']} -> {leg['to_name']}",
                "location": leg["from_name"],
                "reason": f"Weather Advisory ({weather.title()}) on transit corridor",
                "delay_min": weather_pen,
            })
            leg_delay_min += weather_pen

        # Transshipment port dwell congestion
        to_info = leg.get("to_node_info", {})
        if to_info.get("is_seaport") and i < len(legs) - 1:
            congestion_min = 180.0 + (1.0 - carrier_rate) * 300.0
            port_name = to_info.get("port_title") or f"{leg['to_name']} Port"
            stage_delays.append({
                "stage": f"Transshipment Hub Dwell: {leg['to_name']}",
                "location": leg["to_name"],
                "reason": f"Vessel transfer dwell & clearance queue at {port_name}",
                "delay_min": congestion_min,
            })
            delay_markers.append({
                "coords": to_info["coords"],
                "label": f"{port_name} Transfer Congestion",
                "delay_hours": round(congestion_min / 60.0, 1),
            })
            leg_delay_min += congestion_min

        # Import customs & rail transfer delay at destination gateway
        if to_info.get("is_rail_hub") or (to_info.get("is_seaport") and i == len(legs) - 2):
            clearance_min = 240.0 * (1.0 - carrier_rate + 0.3)
            stage_delays.append({
                "stage": f"Import Clearance: {leg['to_name']}",
                "location": leg["to_name"],
                "reason": "Customs clearance & intermodal train manifest queue",
                "delay_min": clearance_min,
            })
            leg_delay_min += clearance_min

    total_delay_min = sum(s["delay_min"] for s in stage_delays)
    total_delay_h = max(1, int(round(total_delay_min / 60.0)))

    total_est_dur_min = total_sched_dur_min + total_delay_min

    # Price formula (Indian Standards: converted to INR at ₹83.5 per USD)
    price_usd = int(round(total_dist_km * cost_per_km_rate + cbm * 45.0 + wm * 30.0 + base_handling_fee))
    price_inr = int(round(price_usd * 83.5, -2))
    price_formatted = format_inr(price_inr)

    # Timeline nodes
    orig_label = origin_node.get("short_name") or origin_node["display_name"].split("(")[0].strip().upper()
    timeline_nodes = [{
        "label": orig_label,
        "full_name": origin_node["display_name"],
        "type": "origin"
    }]
    for i, leg in enumerate(legs):
        transport = leg["mode"]
        clean_label = leg["to_name"].split("(")[0].strip().upper()
        timeline_nodes.append({
            "label": clean_label,
            "full_name": leg["to_name"],
            "type": "destination" if i == len(legs) - 1 else "hub",
            "transport": transport,
        })

    # Waypoints
    waypoints = [{
        "name": origin_node["display_name"],
        "coords": origin_node["coords"],
        "role": "origin",
    }]
    for i, leg in enumerate(legs):
        is_dest = (i == len(legs) - 1)
        to_info = leg.get("to_node_info", {})
        hub_card = None
        if not is_dest and to_info.get("is_seaport"):
            hub_card = {
                "title": to_info.get("port_title") or f"{leg['to_name']} Port",
                "standard_waiting": "24h",
                "estimated_waiting": f"{max(4, int(round(leg['dwell_min'] / 60.0)))}h",
            }

        waypoints.append({
            "name": leg["to_name"],
            "coords": leg["to_coords"],
            "role": "destination" if is_dest else "hub",
            "transport": leg["mode"],
            "hub_card": hub_card,
        })

    # Alert formulation - transparent itemized calculation
    if stage_delays:
        top_stage = stage_delays[0]
        alert_text = (
            f"Expected delay: {total_delay_h}h ({top_stage['reason']}: {int(round(top_stage['delay_min']/60))}h). "
            f"Calculated from intermodal stage transfer and carrier reliability."
        )
    else:
        alert_text = "On-schedule transit projected across all corridor segments."

    # Intermodal transfer count
    transfer_count = max(0, len(legs) - 1)

    return {
        "id": route_id,
        "title": title,
        "route_type": route_type,
        "is_recommended": is_recommended,
        "is_selected": is_selected,
        "timeline_nodes": timeline_nodes,
        "typical_time": format_duration_display(total_sched_dur_min),
        "est_time": format_duration_display(total_est_dur_min),
        "est_price": price_formatted,
        "est_price_inr": price_formatted,
        "est_price_usd": f"${round(price_usd, -2):,}",
        "reliability": f"{int(round(carrier_rate * 100))}%",
        "reliability_num": int(round(carrier_rate * 100)),
        "alert": alert_text,
        "has_alert": bool(stage_delays),
        "badge": badge,
        "distance_km": round(total_dist_km, 0),
        "transfer_count": transfer_count,
        "delay_risk": "High" if total_delay_h > 15 else ("Medium" if total_delay_h > 5 else "Low"),
        "stage_delays": stage_delays,
        "waypoints": waypoints,
        "delay_markers": delay_markers,
        "legs": [
            {
                "from": l["from_name"],
                "to": l["to_name"],
                "mode": l["mode"],
                "distance_km": round(l["dist_km"], 1),
                "scheduled_hours": round(l["sched_dur_min"] / 60.0, 1),
            }
            for l in legs
        ],
        "is_illustrative": is_illustrative,
        "illustrative_note": illustrative_note,
        "sched_duration_minutes": total_sched_dur_min,
        "est_duration_minutes": total_est_dur_min,
        "total_delay_minutes": total_delay_min,
    }
