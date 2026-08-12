"""
SimPy Discrete-Event Simulation Engine for Hospital Queue System
"""

import os
import simpy
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict

from simulator.config import DEPARTMENT_PROFILES, PRIORITY_LEVELS
from simulator.arrival import PatientArrivalEngine
from simulator.queue import PatientEncounterRecord, QueueTracker


class HospitalSimPyEngine:
    def __init__(self, start_date: str = "2026-01-01", sim_days: int = 90, seed: int = 42):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.sim_days = sim_days
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.arrival_engine = PatientArrivalEngine(seed=seed)
        self.patient_counter = 0

    def run_simulation(self) -> pd.DataFrame:
        all_records: List[Dict] = []
        
        for dept_name, dept_config in DEPARTMENT_PROFILES.items():
            print(f"Simulating Department: {dept_name} ({self.sim_days} days)...")
            dept_records = self._simulate_department(dept_name, dept_config)
            all_records.extend(dept_records)

        df = pd.DataFrame(all_records)
        df = df.sort_values(by="arrival_timestamp").reset_index(drop=True)
        return df

    def _simulate_department(self, dept_name: str, dept_config: Dict) -> List[Dict]:
        env = simpy.Environment()
        doctor_resource = simpy.PriorityResource(env, capacity=dept_config["default_doctors"])
        tracker = QueueTracker(department=dept_name)
        
        # SimPy process to generate arrivals per daily shift
        env.process(self._arrival_generator_process(env, dept_name, dept_config, doctor_resource, tracker))
        
        # Simulation duration in minutes: 12 clinic operating hours (720 min) per day
        sim_minutes = self.sim_days * 12 * 60
        env.run(until=sim_minutes)
        
        return [record.to_dict() for record in tracker.completed_encounters]

    def _arrival_generator_process(self, env: simpy.Environment, dept_name: str, dept_config: Dict, doctor_resource: simpy.PriorityResource, tracker: QueueTracker):
        avg_consult = dept_config["avg_consult_time_min"]
        std_consult = dept_config["consult_time_std_dev"]
        
        sigma2 = np.log(1 + (std_consult / avg_consult) ** 2)
        mu = np.log(avg_consult) - 0.5 * sigma2
        sigma = np.sqrt(sigma2)

        while True:
            sim_minute = env.now
            day_idx = int(sim_minute // (12 * 60))
            minute_of_day = int(sim_minute % (12 * 60))
            
            # Map simulation clock to 8:00 AM - 8:00 PM operating day
            current_dt = self.start_date + timedelta(days=day_idx, hours=8, minutes=minute_of_day)
            
            # Generate next inter-arrival interval
            interarrival_min = self.arrival_engine.generate_next_interarrival_time_min(dept_name, current_dt)
            yield env.timeout(interarrival_min)
            
            # If next arrival exceeds the 12-hour clinic day, jump to start of next morning
            if (env.now % (12 * 60)) > 660:  # After 7:00 PM (11th hour)
                next_day_start = (day_idx + 1) * 12 * 60
                delay_to_morning = max(1.0, next_day_start - env.now)
                yield env.timeout(delay_to_morning)
                continue

            self.patient_counter += 1
            pid = f"P{self.patient_counter:06d}"
            
            record = PatientEncounterRecord(patient_id=pid, department=dept_name, arrival_dt=current_dt)
            arrival_type, complexity = self.arrival_engine.sample_patient_attributes(dept_name)
            record.patient_arrival_type = arrival_type
            record.patient_case_complexity = complexity
            
            priority_val = PRIORITY_LEVELS.get(arrival_type, 2)
            
            active_docs = dept_config["default_doctors"]
            break_docs = 1 if (current_dt.hour == 13 and self.rng.random() < 0.4) else 0
            
            tracker.log_arrival(record, current_dt, active_doctors=active_docs, doctors_on_break=break_docs)
            
            env.process(self._patient_journey_process(env, record, priority_val, doctor_resource, tracker, mu, sigma))

    def _patient_journey_process(self, env: simpy.Environment, record: PatientEncounterRecord, priority_val: int, doctor_resource: simpy.PriorityResource, tracker: QueueTracker, mu: float, sigma: float):
        with doctor_resource.request(priority=priority_val) as req:
            yield req
            
            sim_minute = env.now
            day_idx = int(sim_minute // (12 * 60))
            minute_of_day = int(sim_minute % (12 * 60))
            consult_start_dt = self.start_date + timedelta(days=day_idx, hours=8, minutes=minute_of_day)
            
            tracker.log_consultation_start(record, consult_start_dt)
            
            consult_duration_min = self.rng.lognormal(mean=mu, sigma=sigma)
            consult_duration_min = max(3.0, consult_duration_min)
            
            yield env.timeout(consult_duration_min)


if __name__ == "__main__":
    engine = HospitalSimPyEngine(sim_days=90, seed=42)
    df = engine.run_simulation()
    
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "hospital_encounters.csv")
    df.to_csv(csv_path, index=False)
    
    print("\n--- Simulation Complete ---")
    print(f"Total Patient Encounters Generated: {len(df):,}")
    print(f"Saved to: {csv_path}")
