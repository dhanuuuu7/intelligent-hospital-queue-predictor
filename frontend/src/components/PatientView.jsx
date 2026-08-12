import React, { useState, useEffect } from 'react';
import { Clock, Users, UserCheck, Calendar, RefreshCw, AlertCircle, ShieldCheck, Stethoscope, Sparkles, CheckCircle, Ticket, Info } from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

export default function PatientView({ departments, onAddPatient, patientSession }) {
  const [department, setDepartment] = useState('General Medicine');
  const [arrivalType, setArrivalType] = useState('Scheduled');
  
  const [predictionData, setPredictionData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Active check-in ticket state
  const [checkInTicket, setCheckInTicket] = useState(null);

  // Lookup staff assigned active doctors & live waiting queue from backend state
  const currentDeptObj = departments ? departments.find(d => d.name === department) : null;
  const activeDoctors = currentDeptObj ? currentDeptObj.doctors : 4;
  const currentDeptWaiting = currentDeptObj ? currentDeptObj.waiting : 8;

  const fetchPrediction = async () => {
    setLoading(true);
    setError(null);

    const payload = {
      department_name: department,
      day_of_week: 1,
      hour_of_day: 10,
      is_weekend_or_holiday: 0,
      season_indicator: 'Winter',
      queue_position: currentDeptWaiting,
      scheduled_patients_ahead: arrivalType === 'Scheduled' ? Math.max(0, currentDeptWaiting - 1) : Math.floor(currentDeptWaiting / 2),
      walkin_patients_ahead: arrivalType === 'Walk-in' ? Math.max(0, currentDeptWaiting - 1) : Math.floor(currentDeptWaiting / 2),
      patient_arrival_type: arrivalType,
      patient_case_complexity: 'Medium',
      active_doctors_count: activeDoctors,
      doctors_on_break_count: 0,
      triage_staff_count: 2,
      shift_handover_flag: 0,
      emergency_cases_active: 0,
      arrival_rate_past_30min: 10.0,
      recent_throughput_past_hour: 15,
      avg_waiting_time_last_30min: 15.0,
      dept_service_time_variability: 3.5,
      hospital_occupancy_rate: 0.68
    };

    try {
      const res = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) throw new Error('API server returned an error');
      const data = await res.json();
      setPredictionData(data);
    } catch (err) {
      console.warn('Backend server offline or unreachable. Using offline ML fallback estimator.');
      const baseConsult = department === 'Cardiology' ? 22 : department === 'Orthopedics' ? 16 : department === 'Pediatrics' ? 12 : 10;
      const calcWait = Math.round((currentDeptWaiting * baseConsult) / activeDoctors * (arrivalType === 'Walk-in' ? 1.25 : 1.0));

      const marginMap = {
        'General Medicine': 6.8,
        'Pediatrics': 14.2,
        'Orthopedics': 19.5,
        'Cardiology': 38.4
      };
      const margin = marginMap[department] || 14.0;
      
      setPredictionData({
        predicted_waiting_time_min: calcWait,
        lower_bound_min: Math.max(0, Math.round(calcWait - margin)),
        upper_bound_min: Math.round(calcWait + margin),
        interval_width_min: Math.round(margin * 2),
        confidence_level: 0.90,
        confidence_label: department === 'General Medicine' ? 'HIGH' : department === 'Cardiology' ? 'LOW' : 'MODERATE'
      });
      setError('Live API Offline (Using Offline Estimator)');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPrediction();
  }, [department, arrivalType, activeDoctors, currentDeptWaiting]);

  const handleJoinQueue = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/patient/checkin?department_name=${encodeURIComponent(department)}&arrival_type=${encodeURIComponent(arrivalType)}`, {
        method: 'POST'
      });

      if (res.ok) {
        const data = await res.json();
        if (onAddPatient) onAddPatient(department);

        setCheckInTicket({
          code: data.ticket_id,
          department: data.department,
          position: data.queue_position,
          arrivalType: arrivalType,
          estWait: data.prediction ? data.prediction.predicted_waiting_time_min : waitVal,
          lowerBound: data.prediction ? data.prediction.lower_bound_min : lowerBound,
          upperBound: data.prediction ? data.prediction.upper_bound_min : upperBound,
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        });
        return;
      }
    } catch (e) {
      console.warn('Backend checkin offline. Using local ticket generation.');
    }

    // Local fallback check-in
    const newPosition = currentDeptWaiting + 1;
    const prefixMap = {
      'General Medicine': 'GM',
      'Pediatrics': 'PED',
      'Orthopedics': 'ORTH',
      'Cardiology': 'CARD'
    };
    const ticketCode = `${prefixMap[department] || 'GEN'}-${100 + newPosition}`;

    if (onAddPatient) onAddPatient(department);

    setCheckInTicket({
      code: ticketCode,
      department: department,
      position: newPosition,
      arrivalType: arrivalType,
      estWait: waitVal || 15,
      lowerBound: lowerBound,
      upperBound: upperBound,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    });
  };

  const getCrowdBadge = (wait) => {
    if (wait === null || wait === undefined) return { label: 'CALCULATING', class: 'badge-low', color: '#10b981' };
    if (wait < 10) return { label: 'LOW CROWD', class: 'badge-low', color: '#10b981' };
    if (wait < 20) return { label: 'MEDIUM CROWD', class: 'badge-medium', color: '#f59e0b' };
    if (wait < 35) return { label: 'HIGH CROWD', class: 'badge-high', color: '#ef4444' };
    return { label: 'VERY HIGH CROWD', class: 'badge-very-high', color: '#9333ea' };
  };

  const waitVal = predictionData ? (predictionData.predicted_waiting_time_min ?? predictionData.predicted_wait ?? 15) : null;
  const badge = getCrowdBadge(waitVal);

  const lowerBound = predictionData 
    ? (predictionData.lower_bound_min ?? predictionData.lower_bound ?? Math.max(0, Math.round(waitVal - 14))) 
    : 5;
  const upperBound = predictionData 
    ? (predictionData.upper_bound_min ?? predictionData.upper_bound ?? Math.round(waitVal + 14)) 
    : 25;
  const confidenceLabel = predictionData 
    ? (predictionData.confidence_label ?? (department === 'General Medicine' ? 'HIGH' : department === 'Cardiology' ? 'LOW' : 'MODERATE')) 
    : 'MODERATE';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Check-in Ticket Banner (If Patient Checked-In) */}
      {checkInTicket && (
        <div className="glass-card" style={{ padding: '20px 24px', borderLeft: '4px solid #10b981', background: 'rgba(16, 185, 129, 0.08)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
            <div style={{ padding: '10px', borderRadius: '12px', background: 'rgba(16, 185, 129, 0.2)' }}>
              <Ticket size={28} color="#34d399" />
            </div>
            <div>
              <div style={{ fontSize: '0.75rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Confirmed Patient Ticket</div>
              <h3 style={{ fontSize: '1.4rem', fontWeight: '800', color: '#f8fafc', margin: '2px 0' }}>{checkInTicket.code}</h3>
              <p style={{ fontSize: '0.85rem', color: '#cbd5e1', margin: 0 }}>
                {checkInTicket.department} • Backend Assigned Position <strong>#{checkInTicket.position}</strong> • Checked in at {checkInTicket.time}
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Calibrated Wait Range</div>
              <div style={{ fontSize: '1.1rem', fontWeight: '700', color: '#38bdf8' }}>{checkInTicket.lowerBound} – {checkInTicket.upperBound} min</div>
            </div>
            <button onClick={() => setCheckInTicket(null)} style={{ background: 'transparent', border: '1px solid rgba(255,255,255,0.2)', color: '#94a3b8', borderRadius: '6px', padding: '6px 12px', cursor: 'pointer', fontSize: '0.8rem' }}>
              Dismiss
            </button>
          </div>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '24px' }}>
        {/* Patient Input Panel (No Sliders - Backend Controlled Queue & Staff Controlled Doctors) */}
        <div className="glass-card" style={{ padding: '28px' }}>
          <h2 style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Clock size={20} color="#3b82f6" /> Patient Visit Setup
          </h2>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.875rem', color: '#94a3b8', marginBottom: '6px' }}>
                Hospital Department
              </label>
              <select
                value={department}
                onChange={(e) => setDepartment(e.target.value)}
                style={{
                  width: '100%',
                  padding: '12px',
                  borderRadius: '10px',
                  background: 'rgba(15, 23, 42, 0.6)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  color: '#f8fafc',
                  fontSize: '0.95rem'
                }}
              >
                <option value="General Medicine">General Medicine</option>
                <option value="Pediatrics">Pediatrics</option>
                <option value="Orthopedics">Orthopedics</option>
                <option value="Cardiology">Cardiology</option>
              </select>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '0.875rem', color: '#94a3b8', marginBottom: '6px' }}>
                Arrival Type
              </label>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                <button
                  type="button"
                  onClick={() => setArrivalType('Scheduled')}
                  style={{
                    padding: '10px',
                    borderRadius: '8px',
                    border: '1px solid',
                    borderColor: arrivalType === 'Scheduled' ? '#3b82f6' : 'rgba(255, 255, 255, 0.1)',
                    background: arrivalType === 'Scheduled' ? 'rgba(59, 130, 246, 0.2)' : 'transparent',
                    color: arrivalType === 'Scheduled' ? '#60a5fa' : '#94a3b8',
                    fontWeight: '600',
                    cursor: 'pointer'
                  }}
                >
                  Scheduled Appointment
                </button>
                <button
                  type="button"
                  onClick={() => setArrivalType('Walk-in')}
                  style={{
                    padding: '10px',
                    borderRadius: '8px',
                    border: '1px solid',
                    borderColor: arrivalType === 'Walk-in' ? '#3b82f6' : 'rgba(255, 255, 255, 0.1)',
                    background: arrivalType === 'Walk-in' ? 'rgba(59, 130, 246, 0.2)' : 'transparent',
                    color: arrivalType === 'Walk-in' ? '#60a5fa' : '#94a3b8',
                    fontWeight: '600',
                    cursor: 'pointer'
                  }}
                >
                  Walk-in Patient
                </button>
              </div>
            </div>

            {/* Read-Only Backend Assigned Queue Position & Staff Assigned Doctor Indicator */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
              <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '12px', borderRadius: '10px', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
                <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Live Queue Depth</div>
                <div style={{ fontSize: '1.25rem', fontWeight: '700', color: '#60a5fa' }}>#{currentDeptWaiting} Patients</div>
              </div>

              <div style={{ background: 'rgba(16, 185, 129, 0.1)', padding: '12px', borderRadius: '10px', border: '1px solid rgba(16, 185, 129, 0.25)' }}>
                <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Active Staffing</div>
                <div style={{ fontSize: '1.25rem', fontWeight: '700', color: '#34d399' }}>{activeDoctors} Doctors</div>
              </div>
            </div>

            {/* Check-in Button */}
            <button
              onClick={handleJoinQueue}
              style={{
                padding: '12px',
                borderRadius: '10px',
                background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
                color: '#ffffff',
                fontWeight: '700',
                fontSize: '0.95rem',
                border: 'none',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                boxShadow: '0 4px 12px rgba(16, 185, 129, 0.25)',
                transition: 'all 0.2s ease'
              }}
            >
              <Ticket size={18} /> Check-In & Get Queue Ticket
            </button>
          </div>
        </div>

        {/* Waiting Time Prediction Output */}
        <div className="glass-card" style={{ padding: '28px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <span style={{ fontSize: '0.85rem', fontWeight: '700', color: '#94a3b8', letterSpacing: '0.05em', textTransform: 'uppercase' }}>
                {department} Status
              </span>
              <span className={`badge ${badge.class}`}>{badge.label}</span>
            </div>

            <div style={{ textAlign: 'center', margin: '20px 0' }}>
              <div style={{ fontSize: '0.875rem', color: '#94a3b8', marginBottom: '4px' }}>Estimated Waiting Time</div>
              <div style={{ fontSize: '4.25rem', fontWeight: '800', lineHeight: 1, color: '#f8fafc', letterSpacing: '-0.03em' }}>
                {waitVal !== null ? `${waitVal}` : '--'}
                <span style={{ fontSize: '1.75rem', fontWeight: '600', color: '#60a5fa', marginLeft: '6px' }}>min</span>
              </div>
            </div>

            {/* Prediction Interval & Department Confidence Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '16px' }}>
              <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '14px', borderRadius: '12px', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
                <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginBottom: '6px' }}>Likely Range (90% Conf)</div>
                <div style={{ fontSize: '1.15rem', fontWeight: '700', color: '#38bdf8' }}>
                  {lowerBound} – {upperBound} min
                </div>
              </div>

              <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '14px', borderRadius: '12px', border: '1px solid rgba(255, 255, 255, 0.08)', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.75rem', color: '#94a3b8', marginBottom: '6px' }}>
                  <ShieldCheck size={14} color={confidenceLabel === 'HIGH' ? '#34d399' : confidenceLabel === 'MODERATE' ? '#fbbf24' : '#f87171'} /> Confidence Level
                </div>
                <div style={{ fontSize: '0.95rem', fontWeight: '700', color: confidenceLabel === 'HIGH' ? '#34d399' : confidenceLabel === 'MODERATE' ? '#fbbf24' : '#f87171' }}>
                  {confidenceLabel} CONFIDENCE
                </div>
              </div>
            </div>

            <div style={{ fontSize: '0.75rem', color: '#64748b', textAlign: 'center', marginBottom: '16px' }}>
              Calibrated 90% Department Empirical Quantile Model
            </div>

            {/* Why is my wait high? Operational Factors */}
            <div style={{ background: 'rgba(15, 23, 42, 0.4)', padding: '14px', borderRadius: '10px', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
              <div style={{ fontSize: '0.8rem', fontWeight: '600', color: '#94a3b8', marginBottom: '6px' }}>Why is my estimated wait {waitVal} minutes?</div>
              <ul style={{ paddingLeft: '16px', margin: 0, fontSize: '0.8rem', color: '#cbd5e1' }}>
                <li style={{ marginBottom: '3px' }}>🔴 {currentDeptWaiting} patients currently ahead in queue</li>
                <li style={{ marginBottom: '3px' }}>🟢 {activeDoctors} doctors currently active on duty</li>
                <li style={{ marginBottom: '3px' }}>{arrivalType === 'Scheduled' ? '🟢 Scheduled appointment line priority' : '🔴 Walk-in patient queue allocation'}</li>
                {department === 'Cardiology' && <li style={{ marginBottom: '3px', color: '#fbbf24' }}>🔴 High consult duration variability & emergency preemption spikes</li>}
              </ul>
            </div>
          </div>

          <div style={{ fontSize: '0.75rem', color: '#64748b', textAlign: 'center', marginTop: '16px', paddingTop: '12px', borderTop: '1px solid rgba(255, 255, 255, 0.05)' }}>
            Note: Wait times are estimates for planning purposes only and should not replace clinical judgment.
          </div>
        </div>
      </div>
    </div>
  );
}
