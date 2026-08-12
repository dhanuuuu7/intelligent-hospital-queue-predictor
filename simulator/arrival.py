"""
Patient Arrival Engine: Simulates non-homogeneous Poisson arrival process
"""

import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple
from simulator.config import DEPARTMENT_PROFILES, HOURLY_ARRIVAL_MULTIPLIERS, DAY_OF_WEEK_MULTIPLIERS


class PatientArrivalEngine:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

    def get_effective_arrival_rate(self, department: str, timestamp: datetime) -> float:
        """
        Calculates hourly arrival rate lambda(t) based on department baseline,
        time-of-day rush multiplier, and day-of-week multiplier.
        """
        dept_config = DEPARTMENT_PROFILES[department]
        base_rate = dept_config["arrival_rate_per_hour"]
        
        hour = timestamp.hour
        hourly_mult = HOURLY_ARRIVAL_MULTIPLIERS.get(hour, 0.5)
        
        day_of_week = timestamp.weekday()
        day_mult = DAY_OF_WEEK_MULTIPLIERS.get(day_of_week, 1.0)
        
        return max(1.0, base_rate * hourly_mult * day_mult)

    def generate_next_interarrival_time_min(self, department: str, timestamp: datetime) -> float:
        """
        Generates inter-arrival time in minutes using an Exponential distribution (Poisson arrivals).
        """
        rate_per_hour = self.get_effective_arrival_rate(department, timestamp)
        rate_per_minute = rate_per_hour / 60.0
        
        # Exponential distribution: mean interarrival = 1 / lambda
        interarrival_min = self.rng.exponential(1.0 / rate_per_minute)
        return max(0.2, interarrival_min)

    def sample_patient_attributes(self, department: str) -> Tuple[str, str]:
        """
        Samples patient arrival type (Scheduled vs Walk-in vs Emergency) 
        and case complexity based on department profile.
        """
        dept_config = DEPARTMENT_PROFILES[department]
        emerg_prob = dept_config["emergency_prob"]
        
        # 55% Scheduled, 45% Walk-in for non-emergency
        rand_val = self.rng.random()
        if rand_val < emerg_prob:
            arrival_type = "Emergency"
            complexity = "Emergency"
        elif rand_val < emerg_prob + (1.0 - emerg_prob) * 0.55:
            arrival_type = "Scheduled"
            complexity = self._sample_complexity(dept_config["complexity_weights"])
        else:
            arrival_type = "Walk-in"
            complexity = self._sample_complexity(dept_config["complexity_weights"])
            
        return arrival_type, complexity

    def _sample_complexity(self, weights_dict: Dict[str, float]) -> str:
        choices = list(weights_dict.keys())
        probabilities = list(weights_dict.values())
        return self.rng.choice(choices, p=probabilities)
