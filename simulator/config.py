"""
Hospital Environment & Operational Configurations for Discrete-Event Simulation
"""

from typing import Dict, Any

# Department Profiles: Operational parameters tuned for realistic queuing utilization (rho < 0.85)
DEPARTMENT_PROFILES: Dict[str, Dict[str, Any]] = {
    "General Medicine": {
        "avg_consult_time_min": 10.0,
        "consult_time_std_dev": 3.0,
        "default_doctors": 4,           # Max service rate: 24 pts/hr
        "arrival_rate_per_hour": 12.0,  # Base arrival rate
        "emergency_prob": 0.04,
        "complexity_weights": {"Low": 0.45, "Medium": 0.40, "High": 0.11, "Emergency": 0.04}
    },
    "Pediatrics": {
        "avg_consult_time_min": 12.0,
        "consult_time_std_dev": 3.5,
        "default_doctors": 3,           # Max service rate: 15 pts/hr
        "arrival_rate_per_hour": 8.0,   # Base arrival rate
        "emergency_prob": 0.05,
        "complexity_weights": {"Low": 0.40, "Medium": 0.40, "High": 0.15, "Emergency": 0.05}
    },
    "Orthopedics": {
        "avg_consult_time_min": 16.0,
        "consult_time_std_dev": 4.5,
        "default_doctors": 3,           # Max service rate: 11.25 pts/hr
        "arrival_rate_per_hour": 5.5,   # Base arrival rate
        "emergency_prob": 0.06,
        "complexity_weights": {"Low": 0.30, "Medium": 0.48, "High": 0.16, "Emergency": 0.06}
    },
    "Cardiology": {
        "avg_consult_time_min": 20.0,
        "consult_time_std_dev": 5.0,
        "default_doctors": 2,           # Max service rate: 6 pts/hr
        "arrival_rate_per_hour": 3.2,   # Base arrival rate
        "emergency_prob": 0.08,
        "complexity_weights": {"Low": 0.20, "Medium": 0.48, "High": 0.24, "Emergency": 0.08}
    }
}

# Time-of-day arrival rate multipliers (Non-Homogeneous Poisson Process lambda multiplier)
HOURLY_ARRIVAL_MULTIPLIERS: Dict[int, float] = {
    8: 0.7,   # 8 AM - Doors open
    9: 1.3,   # 9 AM - Morning rush starts
    10: 1.5,  # 10 AM - Peak morning volume
    11: 1.4,  # 11 AM - High volume
    12: 0.9,  # 12 PM - Midday decline
    13: 0.5,  # 1 PM - Lunch lull
    14: 1.1,  # 2 PM - Afternoon steady
    15: 1.3,  # 3 PM - Post-lunch rush
    16: 1.0,  # 4 PM - Late afternoon
    17: 0.6,  # 5 PM - Evening taper
    18: 0.3,  # 6 PM - Evening wind-down
    19: 0.1   # 7 PM - Final clinic hours
}

# Day of week arrival multipliers (0=Monday, 6=Sunday)
DAY_OF_WEEK_MULTIPLIERS: Dict[int, float] = {
    0: 1.25,  # Monday surge
    1: 1.05,  # Tuesday busy
    2: 1.00,  # Wednesday average
    3: 0.95,  # Thursday normal
    4: 1.05,  # Friday surge
    5: 0.60,  # Saturday reduced
    6: 0.30   # Sunday minimal
}

# Shift Change Windows (Hour ranges when shift handovers occur)
SHIFT_HANDOVER_HOURS = [13, 14]

# Priority Rankings for Priority Queue (Lower integer = Higher priority)
PRIORITY_LEVELS = {
    "Emergency": 0,
    "Scheduled": 1,
    "Walk-in": 2
}
