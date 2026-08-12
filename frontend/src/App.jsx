import React, { useState, useEffect } from 'react';
import PatientView from './components/PatientView';
import StaffView from './components/StaffView';
import { Activity, User, ShieldCheck, Server, Sparkles, ArrowRight, Lock, Key, CheckCircle, Ticket, LogOut } from 'lucide-react';

const INITIAL_DEPARTMENTS = [
  { name: 'General Medicine', waiting: 8, doctors: 4, avgConsult: 10, estWait: 9, range: '5–14 min', confidence: 'HIGH', crowd: 'LOW', drivers: ['↓ Staff assigned active doctors', '↓ Fast clearance throughput', '↑ 8 Patients ahead'] },
  { name: 'Pediatrics', waiting: 17, doctors: 3, avgConsult: 12, estWait: 18, range: '10–27 min', confidence: 'MODERATE', crowd: 'MEDIUM', drivers: ['↑ Arrival rate > throughput', '↓ Staff assigned active doctors', '↑ 17 Patients ahead'] },
  { name: 'Orthopedics', waiting: 31, doctors: 2, avgConsult: 16, estWait: 34, range: '22–48 min', confidence: 'MODERATE', crowd: 'HIGH', drivers: ['↑ High queue backlog', '↑ Doctor workload pressure', '↓ Staff assigned active doctors'] },
  { name: 'Cardiology', waiting: 42, doctors: 2, avgConsult: 22, estWait: 51, range: '33–71 min', confidence: 'LOW', crowd: 'VERY HIGH', drivers: ['↑ High service variability (sigma=18m)', '↑ Emergency preemption cases', '↑ Severe queue accumulation'] }
];

export default function App() {
  // Navigation states: 'portal' | 'patient-checkin' | 'patient-dashboard' | 'staff-login' | 'staff-dashboard'
  const [viewMode, setViewMode] = useState('portal');
  const [apiStatus, setApiStatus] = useState('checking');
  const [departments, setDepartments] = useState(INITIAL_DEPARTMENTS);

  // Patient Check-in Form state
  const [patientName, setPatientName] = useState('Dhanunjay Nadiminti');
  const [opNumber, setOpNumber] = useState('OP202600123');
  const [selectedDept, setSelectedDept] = useState('Pediatrics');
  const [selectedArrivalType, setSelectedArrivalType] = useState('Scheduled');
  const [patientSession, setPatientSession] = useState(null);

  // Staff Login Form state
  const [staffEmail, setStaffEmail] = useState('staff@hospital.com');
  const [staffPassword, setStaffPassword] = useState('••••••••••••');
  const [staffSession, setStaffSession] = useState(null);
  const [loginError, setLoginError] = useState(null);

  const handleDoctorChange = (index, delta) => {
    setDepartments(prev => {
      const copy = [...prev];
      const newDocs = Math.max(1, copy[index].doctors + delta);
      copy[index].doctors = newDocs;
      
      const newWait = Math.round((copy[index].waiting * copy[index].avgConsult) / newDocs);
      copy[index].estWait = newWait;

      if (newWait < 10) copy[index].crowd = 'LOW';
      else if (newWait < 20) copy[index].crowd = 'MEDIUM';
      else if (newWait < 35) copy[index].crowd = 'HIGH';
      else copy[index].crowd = 'VERY HIGH';

      return copy;
    });
  };

  const handleAddPatient = (deptName) => {
    setDepartments(prev => {
      return prev.map(d => {
        if (d.name === deptName) {
          const newWaiting = d.waiting + 1;
          const newWait = Math.round((newWaiting * d.avgConsult) / d.doctors);
          let newCrowd = 'LOW';
          if (newWait >= 35) newCrowd = 'VERY HIGH';
          else if (newWait >= 20) newCrowd = 'HIGH';
          else if (newWait >= 10) newCrowd = 'MEDIUM';

          return {
            ...d,
            waiting: newWaiting,
            estWait: newWait,
            crowd: newCrowd
          };
        }
        return d;
      });
    });
  };

  const handleCallNextPatient = (index) => {
    setDepartments(prev => {
      const copy = [...prev];
      if (copy[index].waiting > 0) {
        const newWaiting = copy[index].waiting - 1;
        copy[index].waiting = newWaiting;
        const newWait = Math.round((newWaiting * copy[index].avgConsult) / copy[index].doctors);
        copy[index].estWait = newWait;
        if (newWait < 10) copy[index].crowd = 'LOW';
        else if (newWait < 20) copy[index].crowd = 'MEDIUM';
        else if (newWait < 35) copy[index].crowd = 'HIGH';
        else copy[index].crowd = 'VERY HIGH';
      }
      return copy;
    });
  };

  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

  const handlePatientSubmit = async (e) => {
    e.preventDefault();
    if (!patientName.trim() || !opNumber.trim()) return;

    try {
      const res = await fetch(`${API_BASE_URL}/auth/patient-verify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ op_number: opNumber, patient_name: patientName })
      });
      if (res.ok) {
        const data = await res.json();
        handleAddPatient(selectedDept);
        setPatientSession({
          name: data.patient_name,
          opNumber: data.op_number,
          department: selectedDept,
          arrivalType: selectedArrivalType,
          token: data.access_token
        });
        setViewMode('patient-dashboard');
        return;
      }
    } catch (err) {
      console.warn('Backend offline, using local verification fallback.');
    }

    handleAddPatient(selectedDept);
    setPatientSession({
      name: patientName,
      opNumber: opNumber,
      department: selectedDept,
      arrivalType: selectedArrivalType
    });
    setViewMode('patient-dashboard');
  };

  const handleStaffLogin = async (e) => {
    e.preventDefault();
    if (!staffEmail.trim() || !staffPassword.trim()) {
      setLoginError('Please enter valid staff email and password.');
      return;
    }

    setLoginError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: staffEmail, password: staffPassword })
      });

      if (res.ok) {
        const data = await res.json();
        setStaffSession({
          email: data.email,
          name: data.name,
          token: data.access_token
        });
        setViewMode('staff-dashboard');
        return;
      } else {
        const errData = await res.json();
        setLoginError(errData.detail || 'Authentication failed. Check credentials.');
        return;
      }
    } catch (err) {
      console.warn('Backend offline, using fallback staff authorization.');
    }

    setStaffSession({
      email: staffEmail,
      name: 'Dr. Kumar (Staff Operations)'
    });
    setViewMode('staff-dashboard');
  };

  useEffect(() => {
    fetch('http://127.0.0.1:8000/health')
      .then(res => res.json())
      .then(data => {
        if (data.status === 'ok') setApiStatus('online');
        else setApiStatus('offline');
      })
      .catch(() => setApiStatus('offline'));
  }, []);

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '32px 20px' }}>
      {/* Header Bar */}
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '36px', flexWrap: 'wrap', gap: '16px' }}>
        <div style={{ cursor: 'pointer' }} onClick={() => setViewMode('portal')}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{ padding: '8px', borderRadius: '12px', background: 'rgba(59, 130, 246, 0.2)', border: '1px solid rgba(59, 130, 246, 0.3)' }}>
              <Activity size={26} color="#3b82f6" />
            </div>
            <h1 style={{ fontSize: '1.75rem', fontWeight: '700', letterSpacing: '-0.02em', background: 'linear-gradient(135deg, #ffffff 0%, #94a3b8 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              Intelligent Hospital Queue System
            </h1>
          </div>
          <p style={{ fontSize: '0.9rem', color: '#64748b', marginTop: '4px' }}>
            Discrete-Event Simulation & ML Waiting Time Optimization Platform
          </p>
        </div>

        {/* System Status & Portal Reset */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.8rem', padding: '6px 12px', borderRadius: '9999px', background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
            <Server size={14} color={apiStatus === 'online' ? '#10b981' : '#f59e0b'} />
            <span style={{ color: '#94a3b8' }}>
              FastAPI: <strong style={{ color: apiStatus === 'online' ? '#34d399' : '#fbbf24' }}>{apiStatus.toUpperCase()}</strong>
            </span>
          </div>

          {viewMode !== 'portal' && (
            <button
              onClick={() => setViewMode('portal')}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '8px 14px',
                borderRadius: '8px',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                background: 'rgba(15, 23, 42, 0.6)',
                color: '#94a3b8',
                fontWeight: '600',
                fontSize: '0.85rem',
                cursor: 'pointer'
              }}
            >
              <LogOut size={14} /> Hospital Portal Home
            </button>
          )}
        </div>
      </header>

      {/* VIEW 1: ENTRY PORTAL CHOICE SCREEN */}
      {viewMode === 'portal' && (
        <div style={{ maxWidth: '800px', margin: '40px auto 0 auto', textAlign: 'center' }}>
          <div style={{ marginBottom: '32px' }}>
            <h2 style={{ fontSize: '2rem', fontWeight: '800', color: '#f8fafc', marginBottom: '8px' }}>
              🏥 Hospital Entry Portal
            </h2>
            <p style={{ color: '#94a3b8', fontSize: '1rem' }}>
              How would you like to continue today?
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '24px' }}>
            {/* Patient Portal Choice Card */}
            <div 
              className="glass-card" 
              onClick={() => setViewMode('patient-checkin')}
              style={{ 
                padding: '32px 28px', 
                textAlign: 'left', 
                cursor: 'pointer',
                transition: 'transform 0.2s ease, border-color 0.2s ease',
                border: '1px solid rgba(59, 130, 246, 0.25)',
                display: 'flex',
                flexDirection: 'column',
                justify: 'space-between',
                height: '240px'
              }}
            >
              <div>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
                  <div style={{ padding: '12px', borderRadius: '12px', background: 'rgba(59, 130, 246, 0.15)' }}>
                    <User size={30} color="#60a5fa" />
                  </div>
                  <span className="badge badge-low" style={{ fontSize: '0.7rem' }}>PATIENT ACCESS</span>
                </div>
                <h3 style={{ fontSize: '1.3rem', fontWeight: '700', color: '#f8fafc', marginBottom: '6px' }}>
                  👤 Patient Portal
                </h3>
                <p style={{ fontSize: '0.875rem', color: '#94a3b8', lineHeight: 1.4 }}>
                  Enter your OP Number to check live queue status, likely waiting range, and check in.
                </p>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#60a5fa', fontWeight: '700', fontSize: '0.9rem', marginTop: '16px' }}>
                Continue as Patient <ArrowRight size={16} />
              </div>
            </div>

            {/* Staff Portal Choice Card */}
            <div 
              className="glass-card" 
              onClick={() => setViewMode('staff-login')}
              style={{ 
                padding: '32px 28px', 
                textAlign: 'left', 
                cursor: 'pointer',
                transition: 'transform 0.2s ease, border-color 0.2s ease',
                border: '1px solid rgba(16, 185, 129, 0.25)',
                display: 'flex',
                flexDirection: 'column',
                justify: 'space-between',
                height: '240px'
              }}
            >
              <div>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
                  <div style={{ padding: '12px', borderRadius: '12px', background: 'rgba(16, 185, 129, 0.15)' }}>
                    <ShieldCheck size={30} color="#34d399" />
                  </div>
                  <span className="badge badge-low" style={{ fontSize: '0.7rem', background: 'rgba(16, 185, 129, 0.2)', color: '#34d399' }}>STAFF ONLY</span>
                </div>
                <h3 style={{ fontSize: '1.3rem', fontWeight: '700', color: '#f8fafc', marginBottom: '6px' }}>
                  👨‍⚕️ Staff Control Portal
                </h3>
                <p style={{ fontSize: '0.875rem', color: '#94a3b8', lineHeight: 1.4 }}>
                  Hospital staff login for doctor capacity controls, queue management, and what-if simulation.
                </p>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#34d399', fontWeight: '700', fontSize: '0.9rem', marginTop: '16px' }}>
                Staff Login <ArrowRight size={16} />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* VIEW 2: PATIENT IDENTITY CHECK-IN FORM */}
      {viewMode === 'patient-checkin' && (
        <div style={{ maxWidth: '480px', margin: '20px auto 0 auto' }}>
          <div className="glass-card" style={{ padding: '32px' }}>
            <div style={{ textAlign: 'center', marginBottom: '24px' }}>
              <div style={{ display: 'inline-flex', padding: '14px', borderRadius: '16px', background: 'rgba(59, 130, 246, 0.15)', marginBottom: '12px' }}>
                <User size={32} color="#60a5fa" />
              </div>
              <h3 style={{ fontSize: '1.4rem', fontWeight: '700', color: '#f8fafc' }}>Patient Check-In</h3>
              <p style={{ fontSize: '0.875rem', color: '#94a3b8', marginTop: '4px' }}>
                Enter your patient identity details to access live queue status
              </p>
            </div>

            <form onSubmit={handlePatientSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.85rem', color: '#94a3b8', marginBottom: '6px' }}>
                  Patient Full Name
                </label>
                <input
                  type="text"
                  required
                  value={patientName}
                  onChange={(e) => setPatientName(e.target.value)}
                  placeholder="e.g. Dhanunjay Nadiminti"
                  style={{ width: '100%', padding: '12px', borderRadius: '10px', background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(255, 255, 255, 0.1)', color: '#f8fafc', fontSize: '0.95rem' }}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.85rem', color: '#94a3b8', marginBottom: '6px' }}>
                  OP Number (Outpatient Record Key)
                </label>
                <input
                  type="text"
                  required
                  value={opNumber}
                  onChange={(e) => setOpNumber(e.target.value)}
                  placeholder="e.g. OP202600123"
                  style={{ width: '100%', padding: '12px', borderRadius: '10px', background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(255, 255, 255, 0.1)', color: '#f8fafc', fontSize: '0.95rem' }}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.85rem', color: '#94a3b8', marginBottom: '6px' }}>
                  Target Department
                </label>
                <select
                  value={selectedDept}
                  onChange={(e) => setSelectedDept(e.target.value)}
                  style={{ width: '100%', padding: '12px', borderRadius: '10px', background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(255, 255, 255, 0.1)', color: '#f8fafc', fontSize: '0.95rem' }}
                >
                  <option value="General Medicine">General Medicine</option>
                  <option value="Pediatrics">Pediatrics</option>
                  <option value="Orthopedics">Orthopedics</option>
                  <option value="Cardiology">Cardiology</option>
                </select>
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.85rem', color: '#94a3b8', marginBottom: '6px' }}>
                  Arrival Type
                </label>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                  <button
                    type="button"
                    onClick={() => setSelectedArrivalType('Scheduled')}
                    style={{
                      padding: '10px',
                      borderRadius: '8px',
                      border: '1px solid',
                      borderColor: selectedArrivalType === 'Scheduled' ? '#3b82f6' : 'rgba(255, 255, 255, 0.1)',
                      background: selectedArrivalType === 'Scheduled' ? 'rgba(59, 130, 246, 0.2)' : 'transparent',
                      color: selectedArrivalType === 'Scheduled' ? '#60a5fa' : '#94a3b8',
                      fontWeight: '600',
                      cursor: 'pointer'
                    }}
                  >
                    Scheduled
                  </button>
                  <button
                    type="button"
                    onClick={() => setSelectedArrivalType('Walk-in')}
                    style={{
                      padding: '10px',
                      borderRadius: '8px',
                      border: '1px solid',
                      borderColor: selectedArrivalType === 'Walk-in' ? '#3b82f6' : 'rgba(255, 255, 255, 0.1)',
                      background: selectedArrivalType === 'Walk-in' ? 'rgba(59, 130, 246, 0.2)' : 'transparent',
                      color: selectedArrivalType === 'Walk-in' ? '#60a5fa' : '#94a3b8',
                      fontWeight: '600',
                      cursor: 'pointer'
                    }}
                  >
                    Walk-in
                  </button>
                </div>
              </div>

              <button
                type="submit"
                style={{
                  padding: '14px',
                  borderRadius: '10px',
                  background: 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)',
                  color: '#ffffff',
                  fontWeight: '700',
                  fontSize: '1rem',
                  border: 'none',
                  cursor: 'pointer',
                  marginTop: '10px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px'
                }}
              >
                Access Patient Dashboard <ArrowRight size={18} />
              </button>
            </form>
          </div>
        </div>
      )}

      {/* VIEW 3: STAFF CREDENTIAL LOGIN FORM */}
      {viewMode === 'staff-login' && (
        <div style={{ maxWidth: '440px', margin: '20px auto 0 auto' }}>
          <div className="glass-card" style={{ padding: '32px' }}>
            <div style={{ textAlign: 'center', marginBottom: '24px' }}>
              <div style={{ display: 'inline-flex', padding: '14px', borderRadius: '16px', background: 'rgba(16, 185, 129, 0.15)', marginBottom: '12px' }}>
                <ShieldCheck size={32} color="#34d399" />
              </div>
              <h3 style={{ fontSize: '1.4rem', fontWeight: '700', color: '#f8fafc' }}>Staff Credential Login</h3>
              <p style={{ fontSize: '0.875rem', color: '#94a3b8', marginTop: '4px' }}>
                Authorized hospital personnel authentication
              </p>
            </div>

            {loginError && (
              <div style={{ background: 'rgba(239, 68, 68, 0.15)', border: '1px solid rgba(239, 68, 68, 0.3)', color: '#f87171', padding: '10px', borderRadius: '8px', fontSize: '0.85rem', marginBottom: '16px' }}>
                {loginError}
              </div>
            )}

            <form onSubmit={handleStaffLogin} style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.85rem', color: '#94a3b8', marginBottom: '6px' }}>
                  Hospital Staff Email
                </label>
                <input
                  type="email"
                  required
                  value={staffEmail}
                  onChange={(e) => setStaffEmail(e.target.value)}
                  placeholder="staff@hospital.com"
                  style={{ width: '100%', padding: '12px', borderRadius: '10px', background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(255, 255, 255, 0.1)', color: '#f8fafc', fontSize: '0.95rem' }}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.85rem', color: '#94a3b8', marginBottom: '6px' }}>
                  Password
                </label>
                <input
                  type="password"
                  required
                  value={staffPassword}
                  onChange={(e) => setStaffPassword(e.target.value)}
                  placeholder="••••••••••••"
                  style={{ width: '100%', padding: '12px', borderRadius: '10px', background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(255, 255, 255, 0.1)', color: '#f8fafc', fontSize: '0.95rem' }}
                />
              </div>

              <button
                type="submit"
                style={{
                  padding: '14px',
                  borderRadius: '10px',
                  background: 'linear-gradient(135deg, #10b981 0%, #047857 100%)',
                  color: '#ffffff',
                  fontWeight: '700',
                  fontSize: '1rem',
                  border: 'none',
                  cursor: 'pointer',
                  marginTop: '10px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px'
                }}
              >
                Authenticate & Login <Lock size={16} />
              </button>
            </form>
          </div>
        </div>
      )}

      {/* VIEW 4: PATIENT DASHBOARD */}
      {viewMode === 'patient-dashboard' && (
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(15, 23, 42, 0.5)', padding: '12px 20px', borderRadius: '10px', border: '1px solid rgba(255, 255, 255, 0.05)', marginBottom: '24px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '0.9rem', color: '#94a3b8' }}>
              <User size={18} color="#60a5fa" /> Patient: <strong style={{ color: '#f8fafc' }}>{patientSession ? patientSession.name : 'Dhanunjay'}</strong>
              <span style={{ color: '#64748b' }}>|</span> OP Record: <strong style={{ color: '#38bdf8' }}>{patientSession ? patientSession.opNumber : 'OP202600123'}</strong>
            </div>
            <span style={{ fontSize: '0.75rem', padding: '2px 8px', borderRadius: '4px', background: 'rgba(16, 185, 129, 0.2)', color: '#34d399', fontWeight: '700' }}>
              READ-ONLY OBSERVER PORTAL
            </span>
          </div>

          <PatientView 
            departments={departments} 
            onAddPatient={handleAddPatient}
            patientSession={patientSession}
          />
        </div>
      )}

      {/* VIEW 5: STAFF CONTROL DASHBOARD */}
      {viewMode === 'staff-dashboard' && (
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(15, 23, 42, 0.5)', padding: '12px 20px', borderRadius: '10px', border: '1px solid rgba(255, 255, 255, 0.05)', marginBottom: '24px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '0.9rem', color: '#94a3b8' }}>
              <ShieldCheck size={18} color="#34d399" /> Authenticated Staff: <strong style={{ color: '#f8fafc' }}>{staffSession ? staffSession.name : 'Dr. Kumar'}</strong>
              <span style={{ color: '#64748b' }}>|</span> Email: <strong style={{ color: '#60a5fa' }}>{staffSession ? staffSession.email : 'staff@hospital.com'}</strong>
            </div>
            <span style={{ fontSize: '0.75rem', padding: '2px 8px', borderRadius: '4px', background: 'rgba(59, 130, 246, 0.2)', color: '#60a5fa', fontWeight: '700' }}>
              FULL OPERATIONAL CONTROL
            </span>
          </div>

          <StaffView 
            departments={departments} 
            onDoctorChange={handleDoctorChange}
            onCallNext={handleCallNextPatient}
          />
        </div>
      )}

      {/* Footer */}
      <footer style={{ marginTop: '60px', textAlign: 'center', color: '#475569', fontSize: '0.85rem', borderTop: '1px solid rgba(255, 255, 255, 0.05)', paddingTop: '20px' }}>
        Intelligent Hospital Queue & Appointment Optimization System • Role-Based Production System
      </footer>
    </div>
  );
}
