# Data Quality Report

**Total rows**: 200  
**Total columns**: 25  
**Unique shipment IDs**: 190  
**Duplicated IDs (extra rows)**: 10  

## Missing Target / Outcome Values

| Column | Missing Count |
|--------|--------------|
| `actual_departure_datetime` | 55 |
| `actual_arrival_datetime` | 55 |
| `delay_actual_minutes` | 55 |
| `weather_condition_arrival` | 55 |

## Null Counts per Column

| Column | Nulls |
|--------|-------|
| `actual_departure_datetime` | 55 |
| `actual_arrival_datetime` | 55 |
| `origin_state` | 120 |
| `destination_state` | 128 |
| `carrier_name` | 4 |
| `weather_condition_arrival` | 55 |
| `delay_actual_minutes` | 55 |

## Timestamp Ordering Flags (13 issues)

| Index | Shipment ID | Flag |
|-------|-------------|------|
| 3 | SHIP0004 | delay_actual_minutes (197.0) != timestamp-derived (77.0min) |
| 11 | SHIP0012 | delay_actual_minutes (1060.0) != timestamp-derived (80.0min) |
| 22 | SHIP0023 | delay_actual_minutes (221.0) != timestamp-derived (150.0min) |
| 33 | SHIP0034 | delay_actual_minutes (1050.0) != timestamp-derived (165.0min) |
| 60 | SHIP0061 | delay_actual_minutes (233.0) != timestamp-derived (113.0min) |
| 67 | SHIP0068 | delay_actual_minutes (264.0) != timestamp-derived (144.0min) |
| 91 | SHIP0102 | delay_actual_minutes (202.0) != timestamp-derived (24.0min) |
| 103 | SHIP0114 | delay_actual_minutes (143.0) != timestamp-derived (35.0min) |
| 110 | SHIP0121 | delay_actual_minutes (318.0) != timestamp-derived (53.0min) |
| 126 | SHIP0137 | delay_actual_minutes (145.0) != timestamp-derived (45.0min) |
| 133 | SHIP0144 | delay_actual_minutes (98.0) != timestamp-derived (42.0min) |
| 143 | SHIP0154 | delay_actual_minutes (194.0) != timestamp-derived (74.0min) |
| 199 | SHIP0010 | delay_actual_minutes (170.0) != timestamp-derived (54.0min) |

## Duplicated Shipment IDs (20 rows)

IDs: ['SHIP0001', 'SHIP0002', 'SHIP0003', 'SHIP0004', 'SHIP0005', 'SHIP0006', 'SHIP0007', 'SHIP0008', 'SHIP0009', 'SHIP0010']

## Shipment Status Distribution

| Status | Count |
|--------|-------|
| delivered | 121 |
| scheduled | 41 |
| delayed | 24 |
| in_transit | 14 |

## Negative Delays (0 rows)
