"""
Step 9.17: Prediction Latency Performance Benchmark Script
Runs 100 sample prediction iterations and reports latency statistics.
"""

import time
import numpy as np
from fastapi.testclient import TestClient
from app.main import app


def run_latency_benchmark(num_requests: int = 100):
    client = TestClient(app)
    
    payload = {
        "department_name": "General Medicine",
        "day_of_week": 1,
        "hour_of_day": 10,
        "is_weekend_or_holiday": 0,
        "season_indicator": "Winter",
        "queue_position": 8,
        "scheduled_patients_ahead": 5,
        "walkin_patients_ahead": 3,
        "patient_arrival_type": "Scheduled",
        "patient_case_complexity": "Medium",
        "active_doctors_count": 4,
        "doctors_on_break_count": 0,
        "triage_staff_count": 2,
        "shift_handover_flag": 0,
        "emergency_cases_active": 0,
        "arrival_rate_past_30min": 12.0,
        "recent_throughput_past_hour": 18,
        "avg_waiting_time_last_30min": 14.5,
        "dept_service_time_variability": 3.0,
        "hospital_occupancy_rate": 0.70
    }

    # Warmup
    client.post("/predict", json=payload)

    latencies_ms = []

    print(f"\nRunning Latency Benchmark ({num_requests} iterations)...")
    for _ in range(num_requests):
        t0 = time.perf_counter()
        resp = client.post("/predict", json=payload)
        t1 = time.perf_counter()

        assert resp.status_code == 200
        latencies_ms.append((t1 - t0) * 1000.0)

    avg_lat = np.mean(latencies_ms)
    med_lat = np.median(latencies_ms)
    p95_lat = np.percentile(latencies_ms, 95)
    min_lat = np.min(latencies_ms)
    max_lat = np.max(latencies_ms)

    print("=" * 60)
    print("PREDICTION LATENCY BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total Requests : {num_requests}")
    print(f"Average Latency: {avg_lat:.2f} ms")
    print(f"Median Latency : {med_lat:.2f} ms")
    print(f"P95 Latency    : {p95_lat:.2f} ms")
    print(f"Min Latency    : {min_lat:.2f} ms")
    print(f"Max Latency    : {max_lat:.2f} ms")
    print("=" * 60)

    return {
        "num_requests": num_requests,
        "avg_ms": round(avg_lat, 2),
        "median_ms": round(med_lat, 2),
        "p95_ms": round(p95_lat, 2),
        "min_ms": round(min_lat, 2),
        "max_ms": round(max_lat, 2)
    }


if __name__ == "__main__":
    run_latency_benchmark()
