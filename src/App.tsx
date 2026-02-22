import React, { useState, useEffect, useMemo } from 'react';
import { Activity, Play, Pause, RotateCcw } from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceDot,
  ScatterChart, Scatter
} from 'recharts';

// --- ODE Solver (Runge-Kutta 4th Order) ---
function solveODE(m, k, gammaFunc, y0, v0, tMax, dt) {
  let y = y0;
  let v = v0;
  let t = 0;
  const data = [{ t, y, v }];

  const steps = Math.ceil(tMax / dt);
  for (let i = 0; i < steps; i++) {
    const k1y = v;
    const k1v = (-gammaFunc(y, v) * v - k * y) / m;

    const y2 = y + 0.5 * dt * k1y;
    const v2 = v + 0.5 * dt * k1v;
    const k2y = v2;
    const k2v = (-gammaFunc(y2, v2) * v2 - k * y2) / m;

    const y3 = y + 0.5 * dt * k2y;
    const v3 = v + 0.5 * dt * k2v;
    const k3y = v3;
    const k3v = (-gammaFunc(y3, v3) * v3 - k * y3) / m;

    const y4 = y + dt * k3y;
    const v4 = v + dt * k3v;
    const k4y = v4;
    const k4v = (-gammaFunc(y4, v4) * v4 - k * y4) / m;

    y = y + (dt / 6) * (k1y + 2 * k2y + 2 * k3y + k4y);
    v = v + (dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v);
    t = t + dt;

    data.push({ t, y, v });
  }
  return data;
}

// --- Helper for drawing the spring ---
function drawSpring(x, yStart, yEnd, coils, width) {
  const height = yEnd - yStart;
  const coilHeight = height / coils;
  let path = `M ${x} ${yStart} `;
  for (let i = 0; i < coils; i++) {
    path += `L ${x - width / 2} ${yStart + coilHeight * (i + 0.25)} `;
    path += `L ${x + width / 2} ${yStart + coilHeight * (i + 0.75)} `;
    path += `L ${x} ${yStart + coilHeight * (i + 1)} `;
  }
  return path;
}

// --- UI Components ---
const Slider = ({ label, value, min, max, step, onChange }) => (
  <div className="flex flex-col gap-2">
    <div className="flex justify-between items-center">
      <label className="text-sm font-medium text-slate-700">{label}</label>
      <span className="text-xs font-mono text-slate-600 bg-slate-100 px-2 py-1 rounded-md border border-slate-200">
        {value.toFixed(step < 0.1 ? 2 : 1)}
      </span>
    </div>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600 hover:accent-blue-700 transition-all"
    />
  </div>
);

const Select = ({ label, value, options, onChange }) => (
  <div className="flex flex-col gap-2">
    <label className="text-sm font-medium text-slate-700">{label}</label>
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full bg-slate-50 border border-slate-300 text-slate-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block p-2.5 outline-none transition-colors"
    >
      {options.map((opt) => (
        <option key={opt.value} value={opt.value}>
          {opt.label}
        </option>
      ))}
    </select>
  </div>
);

const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-slate-800 text-white text-xs rounded-lg py-2 px-3 shadow-xl border border-slate-700">
        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
          <span className="text-slate-400">Time (t):</span>
          <span className="font-mono text-right">{data.t.toFixed(2)}</span>
          <span className="text-slate-400">Position (y):</span>
          <span className="font-mono text-right">{data.y.toFixed(4)}</span>
          <span className="text-slate-400">Velocity (y'):</span>
          <span className="font-mono text-right">{data.v.toFixed(4)}</span>
        </div>
      </div>
    );
  }
  return null;
};

const PhysicalModel = ({ currentY, scale = 200 }) => {
  const groundY = 280;
  const massBaseY = 150 - currentY * scale;

  // Spring
  const springX = 100;
  const springWidth = 30;
  const springCoils = 12;
  const springPath = drawSpring(springX, massBaseY, groundY, springCoils, springWidth);

  // Damper
  const damperX = 200;
  const cylinderHeight = 80;
  const cylinderWidth = 24;
  const pistonWidth = 6;

  return (
    <svg width="100%" height="100%" viewBox="0 0 300 320" className="bg-transparent">
      {/* Ground */}
      <line x1="40" y1={groundY} x2="260" y2={groundY} stroke="#64748b" strokeWidth="4" strokeLinecap="round" />
      <line x1="40" y1={groundY + 8} x2="260" y2={groundY + 8} stroke="#cbd5e1" strokeWidth="2" strokeLinecap="round" />

      {/* Bases */}
      <rect x={springX - 20} y={groundY - 4} width={40} height={4} fill="#475569" />
      <rect x={damperX - 20} y={groundY - 4} width={40} height={4} fill="#475569" />

      {/* Spring */}
      <path d={springPath} fill="none" stroke="#3b82f6" strokeWidth="3" strokeLinejoin="round" />

      {/* Damper Cylinder (Ground attached) */}
      <rect x={damperX - cylinderWidth / 2} y={groundY - cylinderHeight} width={cylinderWidth} height={cylinderHeight} fill="#ef4444" rx="2" />
      <rect x={damperX - cylinderWidth / 2 + 4} y={groundY - cylinderHeight} width={cylinderWidth - 8} height={cylinderHeight} fill="#dc2626" rx="2" />

      {/* Damper Piston (Mass attached) */}
      <line x1={damperX} y1={massBaseY} x2={damperX} y2={groundY - cylinderHeight + 20} stroke="#78350f" strokeWidth={pistonWidth} />
      <line x1={damperX - 12} y1={groundY - cylinderHeight + 20} x2={damperX + 12} y2={groundY - cylinderHeight + 20} stroke="#78350f" strokeWidth="4" strokeLinecap="round" />

      {/* Mass (L-shape) */}
      <path d={`M 80 ${massBaseY} L 220 ${massBaseY} L 220 ${massBaseY - 120} L 200 ${massBaseY - 120} L 200 ${massBaseY - 20} L 80 ${massBaseY - 20} Z`} fill="#2563eb" />

      {/* Connection points */}
      <circle cx={springX} cy={massBaseY} r="4" fill="#1e293b" />
      <circle cx={damperX} cy={massBaseY} r="4" fill="#1e293b" />
    </svg>
  );
};

// --- Main App Component ---
export default function App() {
  const [gammaType, setGammaType] = useState('exp_y');
  const [y0, setY0] = useState(0.1);
  const [v0, setV0] = useState(0.2);
  const [m, setM] = useState(0.1);
  const [k, setK] = useState(1.0);
  const [tMax, setTMax] = useState(35);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);

  const odeData = useMemo(() => {
    let gammaFunc;
    switch (gammaType) {
      case 'exp_y':
        gammaFunc = (y, v) => 1 - Math.exp(-10 * y * y);
        break;
      case 'exp_v':
        gammaFunc = (y, v) => 1 - Math.exp(-10 * v * v);
        break;
      case 'vdp':
        gammaFunc = (y, v) => 0.5 * (y * y - 1);
        break;
      case 'constant':
        gammaFunc = (y, v) => 0.2;
        break;
      default:
        gammaFunc = (y, v) => 1 - Math.exp(-10 * y * y);
    }
    return solveODE(m, k, gammaFunc, y0, v0, tMax, 0.05);
  }, [gammaType, y0, v0, m, k, tMax]);

  useEffect(() => {
    if (!isPlaying) return;

    let animationFrameId;
    let lastTime = performance.now();

    const animate = (time) => {
      const dt = (time - lastTime) / 1000;
      lastTime = time;

      setCurrentTime((prev) => {
        const next = prev + dt * 2; // 2x speed
        return next > tMax ? 0 : next;
      });

      animationFrameId = requestAnimationFrame(animate);
    };

    animationFrameId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationFrameId);
  }, [tMax, isPlaying]);

  const currentIndex = Math.min(
    Math.floor((currentTime / tMax) * (odeData.length - 1)),
    odeData.length - 1
  );
  const currentData = odeData[currentIndex] || odeData[0];

  return (
    <div className="flex flex-col md:flex-row h-screen bg-slate-50 text-slate-900 font-sans overflow-hidden">
      {/* Sidebar */}
      <div className="w-full md:w-80 bg-white border-b md:border-r border-slate-200 p-6 flex flex-col gap-6 overflow-y-auto shadow-sm z-10 shrink-0">
        <div className="flex items-center gap-3 pb-4 border-b border-slate-100">
          <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white font-bold shadow-md shadow-blue-200">
            <Activity size={18} />
          </div>
          <h1 className="text-lg font-semibold text-slate-800">Active Shock Absorber</h1>
        </div>

        <div className="flex flex-col gap-5">
          <Select
            label="Damping Function γ"
            value={gammaType}
            onChange={setGammaType}
            options={[
              { value: 'exp_y', label: '1 - exp(-10 y²)' },
              { value: 'exp_v', label: '1 - exp(-10 y\'²)' },
              { value: 'vdp', label: '0.5 (y² - 1) [Van der Pol]' },
              { value: 'constant', label: '0.2 [Constant]' },
            ]}
          />
          <Slider label="Initial Position (y₀)" value={y0} min={-0.5} max={0.5} step={0.01} onChange={setY0} />
          <Slider label="Initial Velocity (y'₀)" value={v0} min={-1.0} max={1.0} step={0.01} onChange={setV0} />
          <Slider label="Mass (m)" value={m} min={0.01} max={1.0} step={0.01} onChange={setM} />
          <Slider label="Spring Constant (k)" value={k} min={0.1} max={5.0} step={0.1} onChange={setK} />
          <Slider label="Simulation Time (t)" value={tMax} min={10} max={100} step={1} onChange={setTMax} />
        </div>

        <div className="mt-auto pt-4 border-t border-slate-100 flex gap-2">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="flex-1 bg-slate-100 hover:bg-slate-200 text-slate-700 font-medium py-2.5 px-4 rounded-lg flex items-center justify-center gap-2 transition-colors"
          >
            {isPlaying ? <Pause size={16} /> : <Play size={16} />}
            {isPlaying ? 'Pause' : 'Play'}
          </button>
          <button
            onClick={() => {
              setCurrentTime(0);
              setIsPlaying(true);
            }}
            className="bg-slate-100 hover:bg-slate-200 text-slate-700 font-medium py-2.5 px-4 rounded-lg flex items-center justify-center transition-colors"
            title="Reset"
          >
            <RotateCcw size={16} />
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 p-4 md:p-6 flex flex-col gap-4 md:gap-6 overflow-y-auto">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 md:gap-6 min-h-[400px]">
          {/* Physical Model */}
          <div className="flex flex-col gap-2">
            <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Physical System</h2>
            <div className="flex-1 bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden flex items-center justify-center p-4">
              <PhysicalModel currentY={currentData.y} />
            </div>
          </div>

          {/* Phase Portrait */}
          <div className="flex flex-col gap-2">
            <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Phase Portrait y'(t) vs y(t)</h2>
            <div className="flex-1 bg-white rounded-2xl shadow-sm border border-slate-200 p-4">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis type="number" dataKey="y" name="y" domain={['auto', 'auto']} tick={{ fontSize: 12, fill: '#64748b' }} tickFormatter={(v) => v.toFixed(2)} />
                  <YAxis type="number" dataKey="v" name="y'" domain={['auto', 'auto']} tick={{ fontSize: 12, fill: '#64748b' }} tickFormatter={(v) => v.toFixed(2)} />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} content={CustomTooltip} />
                  <Scatter data={odeData} fill="#0ea5e9" line={{ stroke: '#0ea5e9', strokeWidth: 2 }} shape={() => null} isAnimationActive={false} />
                  <Scatter data={[currentData]} fill="#ef4444" shape="circle" isAnimationActive={false} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Time Series */}
        <div className="flex flex-col gap-2 flex-1 min-h-[300px]">
          <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Time Series y(t)</h2>
          <div className="flex-1 bg-white rounded-2xl shadow-sm border border-slate-200 p-4">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={odeData} margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis type="number" dataKey="t" name="t" domain={[0, tMax]} tick={{ fontSize: 12, fill: '#64748b' }} />
                <YAxis type="number" dataKey="y" name="y" domain={['auto', 'auto']} tick={{ fontSize: 12, fill: '#64748b' }} tickFormatter={(v) => v.toFixed(2)} />
                <Tooltip content={CustomTooltip} />
                <Line type="monotone" dataKey="y" stroke="#0ea5e9" dot={false} strokeWidth={2} isAnimationActive={false} />
                <ReferenceDot x={currentData.t} y={currentData.y} r={6} fill="#ef4444" stroke="#fff" strokeWidth={2} isFront={true} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
