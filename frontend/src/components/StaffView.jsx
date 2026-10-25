import React, { useState, useEffect } from 'react';
import { AlertTriangle, Sparkles, Info, UserMinus, UserPlus, Sliders, Activity, TrendingDown, TrendingUp, ShieldAlert, Cpu, BarChart3, LineChart, CheckCircle2, Building2, Send, RefreshCw } from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

export default function StaffView({ departments, onDoctorChange, onCallNext }) {
  const [selectedSimDept, setSelectedSimDept] = useState('Pediatrics');
  const [doctorDelta, setDoctorDelta] = useState(1);
  const [emergencyDelta, setEmergencyDelta] = useState(0);
  const [whatIfResult, setWhatIfResult] = useState(null);
  const [simLoading, setSimLoading] = useState(false);

  // Phase 12 & 13 Enterprise State
  const [forecastData, setForecastData] = useState(null);
  const [aiRecommendation, setAiRecommendation] = useState(null);
  const [feedbackMetrics, setFeedbackMetrics] = useState(null);
  const [fleetStatus, setFleetStatus] = useState(null);
  const [retrainStatus, setRetrainStatus] = useState(null);
  const [retrainLoading, setRetrainLoading] = useState(false);
  const [sentAlerts, setSentAlerts] = useState([]);

  const fetchOperationsIntelligence = async () => {
    try {
      const fRes = await fetch(`${API_BASE_URL}/analytics/forecast/${encodeURIComponent(selectedSimDept)}`);
      if (fRes.ok) setForecastData(await fRes.json());

      const rRes = await fetch(`${API_BASE_URL}/analytics/recommendation/${encodeURIComponent(selectedSimDept)}`);
      if (rRes.ok) setAiRecommendation(await rRes.json());

      const mRes = await fetch(`${API_BASE_URL}/feedback/metrics`);
      if (mRes.ok) setFeedbackMetrics(await mRes.json());

      const flRes = await fetch(`${API_BASE_URL}/fleet/network-status`);
      if (flRes.ok) setFleetStatus(await flRes.json());

      const nRes = await fetch(`${API_BASE_URL}/notifications/logs`);
      if (nRes.ok) setSentAlerts(await nRes.json());
    } catch (e) {
      console.warn('Phase 12/13 analytics offline fallback.');
    }
  };

  useEffect(() => {
    fetchOperationsIntelligence();
  }, [selectedSimDept, departments]);

  const handleTriggerAutoRetrain = async () => {
    setRetrainLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/mlops/auto-retrain`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setRetrainStatus(data);
      }
    } catch (e) {
      console.warn('Auto-retrain offline fallback.');
    } finally {
      setRetrainLoading(false);
    }
  };

  const runWhatIfSimulation = async () => {
    setSimLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/staff/what-if?department_name=${encodeURIComponent(selectedSimDept)}&doctor_delta=${doctorDelta}&emergency_delta=${emergencyDelta}`, {
        method: 'POST'
      });
      if (res.ok) {
        const data = await res.json();
        setWhatIfResult(data);
        setSimLoading(false);
        return;
      }
    } catch (e) {
      console.warn('Backend What-If simulation offline. Running local calculation.');
    }

    const dept = departments ? departments.find(d => d.name === selectedSimDept) : null;
    const currDocs = dept ? dept.doctors : 3;
    const currWait = dept ? dept.estWait : 25;
    const simDocs = Math.max(1, currDocs + doctorDelta);
    const simWait = Math.max(5, Math.round((currWait * currDocs) / simDocs + (emergencyDelta * 8)));
    const diff = simWait - currWait;

    setWhatIfResult({
      department: selectedSimDept,
      current_doctors: currDocs,
      simulated_doctors: simDocs,
      current_predicted_wait_min: currWait,
      simulated_predicted_wait_min: simWait,
      wait_time_delta_min: diff,
      impact_summary: diff < 0 
        ? `Adding ${doctorDelta} doctor(s) reduces wait time from ${currWait} min → ${simWait} min (↓ ${Math.abs(diff)} min reduction).`
        : `Simulated wait increases from ${currWait} min → ${simWait} min (↑ ${diff} min impact).`
    });
    setSimLoading(false);
  };

  const getCrowdBadge = (crowd) => {
    switch (crowd) {
      case 'LOW': return { label: '🟢 LOW', class: 'badge-low' };
      case 'MEDIUM': return { label: '🟡 MEDIUM', class: 'badge-medium' };
      case 'HIGH': return { label: '🔴 HIGH', class: 'badge-high' };
      case 'VERY HIGH': return { label: '🚨 VERY HIGH', class: 'badge-very-high' };
      default: return { label: crowd, class: 'badge-low' };
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Operational Alert Banner */}
      <div 
        className="glass-card" 
        style={{ 
          padding: '20px 24px', 
          borderLeft: '4px solid #10b981', 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between',
          background: 'rgba(16, 185, 129, 0.08)'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Sparkles size={24} color="#34d399" />
          <div>
            <h4 style={{ fontWeight: '600', color: '#f8fafc' }}>Phase 13 Enterprise Fleet & Self-Healing MLOps Platform Active</h4>
            <p style={{ fontSize: '0.875rem', color: '#94a3b8' }}>
              Multi-hospital fleet network, WebSockets real-time sync, SMS dispatcher, and auto-retraining pipeline online.
            </p>
          </div>
        </div>
        <button
          onClick={handleTriggerAutoRetrain}
          disabled={retrainLoading}
          style={{
            padding: '8px 14px',
            borderRadius: '8px',
            background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
            color: 'white',
            fontWeight: '700',
            fontSize: '0.85rem',
            border: 'none',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}
        >
          <RefreshCw size={14} className={retrainLoading ? 'spin' : ''} /> {retrainLoading ? 'Retraining...' : 'Trigger MLOps Self-Heal'}
        </button>
      </div>

      {/* Retrain Status Banner */}
      {retrainStatus && (
        <div style={{ background: 'rgba(59, 130, 246, 0.15)', border: '1px solid rgba(59, 130, 246, 0.3)', padding: '14px', borderRadius: '10px', color: '#38bdf8', fontSize: '0.9rem' }}>
          <strong>MLOps Retrain Result ({retrainStatus.status}):</strong> {retrainStatus.action} • Test MAE: {retrainStatus.test_mae} min • Test R²: {retrainStatus.test_r2}
        </div>
      )}

      {/* Operational Cards Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px' }}>
        {departments && departments.map((dept, idx) => {
          const badge = getCrowdBadge(dept.crowd);
          return (
            <div key={dept.name} className="glass-card" style={{ padding: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <h3 style={{ fontSize: '1.1rem', fontWeight: '600' }}>{dept.name}</h3>
                <span className={`badge ${badge.class}`}>{badge.label}</span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', margin: '16px 0' }}>
                <div style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '12px', borderRadius: '10px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                    <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Patients Waiting</span>
                    {onCallNext && (
                      <button
                        onClick={() => onCallNext(idx)}
                        title="Call next waiting patient"
                        style={{
                          fontSize: '0.7rem',
                          padding: '2px 6px',
                          borderRadius: '4px',
                          border: '1px solid rgba(16, 185, 129, 0.4)',
                          background: 'rgba(16, 185, 129, 0.15)',
                          color: '#34d399',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '2px'
                        }}
                      >
                        <UserMinus size={12} /> Call Next
                      </button>
                    )}
                  </div>
                  <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#f8fafc' }}>{dept.waiting}</div>
                </div>

                <div style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '12px', borderRadius: '10px' }}>
                  <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginBottom: '4px' }}>Predicted Wait</div>
                  <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#60a5fa' }}>{dept.estWait} min</div>
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: '12px', borderTop: '1px solid rgba(255, 255, 255, 0.08)' }}>
                <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>Doctors Active: <strong>{dept.doctors}</strong></span>
                <div style={{ display: 'flex', gap: '6px' }}>
                  <button
                    onClick={() => onDoctorChange && onDoctorChange(idx, -1)}
                    style={{
                      width: '28px',
                      height: '28px',
                      borderRadius: '6px',
                      border: '1px solid rgba(255, 255, 255, 0.15)',
                      background: 'rgba(255, 255, 255, 0.05)',
                      color: 'white',
                      fontWeight: 'bold',
                      cursor: 'pointer'
                    }}
                  >
                    -
                  </button>
                  <button
                    onClick={() => onDoctorChange && onDoctorChange(idx, 1)}
                    style={{
                      width: '28px',
                      height: '28px',
                      borderRadius: '6px',
                      border: '1px solid rgba(59, 130, 246, 0.4)',
                      background: 'rgba(59, 130, 246, 0.2)',
                      color: '#60a5fa',
                      fontWeight: 'bold',
                      cursor: 'pointer'
                    }}
                  >
                    +
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* PHASE 13: Multi-Hospital Fleet Network & Position SMS Log */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px' }}>
        {/* Fleet Network Status Card */}
        <div className="glass-card" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: '600', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Building2 size={20} color="#34d399" /> Multi-Hospital Fleet Network Capacity
          </h3>

          {fleetStatus ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {fleetStatus.map((fac) => (
                <div key={fac.facility_id} style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '12px 14px', borderRadius: '10px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', border: '1px solid rgba(255,255,255,0.05)' }}>
                  <div>
                    <div style={{ fontWeight: '600', color: '#f8fafc', fontSize: '0.9rem' }}>{fac.name}</div>
                    <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Distance: {fac.distance_km} km • Total Waiting: {fac.total_waiting}</div>
                  </div>
                  <span className={`badge ${fac.load_status === 'OPTIMAL' ? 'badge-low' : 'badge-high'}`}>
                    {fac.avg_wait_min}m wait
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ color: '#94a3b8', fontSize: '0.9rem' }}>Loading regional fleet data...</div>
          )}
        </div>

        {/* Patient SMS Alert Log Card */}
        <div className="glass-card" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: '600', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Send size={20} color="#60a5fa" /> Automated Patient SMS Alerts Log (Position ≤ #3)
          </h3>

          {sentAlerts && sentAlerts.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {sentAlerts.map((alt) => (
                <div key={alt.id} style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '10px 12px', borderRadius: '8px', borderLeft: '3px solid #3b82f6' }}>
                  <div style={{ fontSize: '0.75rem', color: '#94a3b8', display: 'flex', justifyContent: 'space-between' }}>
                    <span>To: <strong>{alt.patient_name}</strong> ({alt.op_number})</span>
                    <span style={{ color: '#34d399' }}>{alt.status}</span>
                  </div>
                  <div style={{ fontSize: '0.8rem', color: '#cbd5e1', marginTop: '4px' }}>
                    {alt.message}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ color: '#94a3b8', fontSize: '0.85rem', textAlign: 'center', padding: '20px 0' }}>
              Automated SMS alerts dispatch when a patient's backend queue position reaches #3.
            </div>
          )}
        </div>
      </div>

      {/* Staff Decision Support "What-If" Scenario Simulator */}
      <div className="glass-card" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: '600', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Sliders size={20} color="#3b82f6" /> Staff Decision Support: "What-If" Scenario Simulator
        </h3>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            <div>
              <label style={{ fontSize: '0.85rem', color: '#94a3b8', display: 'block', marginBottom: '4px' }}>Target Department</label>
              <select
                value={selectedSimDept}
                onChange={(e) => setSelectedSimDept(e.target.value)}
                style={{ width: '100%', padding: '10px', borderRadius: '8px', background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(255, 255, 255, 0.1)', color: 'white' }}
              >
                <option value="General Medicine">General Medicine</option>
                <option value="Pediatrics">Pediatrics</option>
                <option value="Orthopedics">Orthopedics</option>
                <option value="Cardiology">Cardiology</option>
              </select>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
              <div>
                <label style={{ fontSize: '0.85rem', color: '#94a3b8', display: 'block', marginBottom: '4px' }}>Add Doctors</label>
                <select
                  value={doctorDelta}
                  onChange={(e) => setDoctorDelta(parseInt(e.target.value))}
                  style={{ width: '100%', padding: '10px', borderRadius: '8px', background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(255, 255, 255, 0.1)', color: 'white' }}
                >
                  <option value={1}>+1 Doctor</option>
                  <option value={2}>+2 Doctors</option>
                  <option value={-1}>-1 Doctor</option>
                </select>
              </div>

              <div>
                <label style={{ fontSize: '0.85rem', color: '#94a3b8', display: 'block', marginBottom: '4px' }}>Emergency Spike</label>
                <select
                  value={emergencyDelta}
                  onChange={(e) => setEmergencyDelta(parseInt(e.target.value))}
                  style={{ width: '100%', padding: '10px', borderRadius: '8px', background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(255, 255, 255, 0.1)', color: 'white' }}
                >
                  <option value={0}>0 Emergency Cases</option>
                  <option value={1}>+1 Emergency Case</option>
                  <option value={3}>+3 Emergency Cases</option>
                </select>
              </div>
            </div>

            <button
              onClick={runWhatIfSimulation}
              disabled={simLoading}
              style={{
                padding: '12px',
                borderRadius: '8px',
                background: 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)',
                color: 'white',
                fontWeight: '700',
                border: 'none',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px'
              }}
            >
              <Activity size={18} /> Run Scenario Simulation
            </button>
          </div>

          {/* Simulation Output Card */}
          <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '20px', borderRadius: '12px', border: '1px solid rgba(255, 255, 255, 0.08)', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            {whatIfResult ? (
              <div>
                <div style={{ fontSize: '0.8rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '6px' }}>
                  Simulated Outcome ({whatIfResult.department})
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', margin: '12px 0' }}>
                  <div style={{ fontSize: '1.2rem', color: '#94a3b8' }}>
                    Current: <strong>{whatIfResult.current_predicted_wait_min}m</strong>
                  </div>
                  <div style={{ fontSize: '1.5rem', fontWeight: '800', color: whatIfResult.wait_time_delta_min <= 0 ? '#34d399' : '#f87171' }}>
                    ➜ {whatIfResult.simulated_predicted_wait_min} min
                  </div>
                </div>
                <div style={{ fontSize: '0.9rem', color: whatIfResult.wait_time_delta_min <= 0 ? '#34d399' : '#f87171', fontWeight: '600', background: 'rgba(15, 23, 42, 0.8)', padding: '10px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                  {whatIfResult.impact_summary}
                </div>
              </div>
            ) : (
              <div style={{ textAlign: 'center', color: '#64748b', fontSize: '0.9rem' }}>
                Select capacity changes and click <strong>Run Scenario Simulation</strong> to test wait time impact.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
