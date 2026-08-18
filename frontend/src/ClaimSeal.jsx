import React, { useState, useEffect, useCallback, useRef, createContext, useContext } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell
} from "recharts";
import {
  LogOut, LayoutDashboard, FileText, Shield, ShieldCheck, ShieldAlert,
  BarChart2, Search, RefreshCw, AlertTriangle, CheckCircle2, XCircle,
  Clock, Eye, EyeOff, Upload, X, Menu, AlertOctagon,
  Car, Info, ChevronRight
} from "lucide-react";

/* ─── Auth helpers ───────────────────────────────────────────── */
const BASE = "/api/v1";
const getToken = () => sessionStorage.getItem("jwt_access");
const setToken = (t) => sessionStorage.setItem("jwt_access", t);
const clearToken = () => sessionStorage.removeItem("jwt_access");
const AUTH_TOKEN_HEADER = () => ({ Authorization: `Bearer ${getToken()}` });

async function apiFetch(path, opts = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...AUTH_TOKEN_HEADER(), ...(opts.headers || {}) },
    ...opts,
  });
  const json = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(json?.error?.message || `HTTP ${res.status}`);
  return json.data ?? json;
}

function mapClaim(c, userName) {
  const statusMap = { SUBMITTED: "Pending", ANALYZED: "Pending", UNDER_REVIEW: "Under Review", APPROVED: "Approved", REJECTED: "Rejected", ESCALATED: "Escalated" };
  const isFraud = c.prediction === "Fraud" ? true : c.prediction === "Non-Fraud" ? false : null;
  const accuracy = c.fraud_probability != null
    ? +(Math.max(c.fraud_probability, c.non_fraud_probability) * 100).toFixed(1)
    : null;
  const approvalPct = c.non_fraud_probability != null ? Math.round(c.non_fraud_probability * 100) : null;
  return {
    id: c.id,
    ref: c.claim_reference || c.id,
    claimantName: userName || c.user_id,
    policyNumber: c.vehicle_number,
    vehicleModel: c.vehicle_model,
    vehicleYear: c.vehicle_year,
    incidentDate: c.incident_date,
    description: c.description,
    estimatedCost: c.claim_amount,
    imageUrl: c.image_url || null,
    status: statusMap[c.status] || c.status,
    rawStatus: c.status,
    isFraud,
    accuracy,
    approvalPct,
    riskLevel: c.risk_level || null,
    recommendation: c.recommendation || null,
  };
}

let _userName = "";

const api = {
  loginUser: async (email, password, role) => {
    if (!email || !password) throw new Error("Email and password are required.");
    let data;
    try {
      data = await apiFetch("/auth/login", { method: "POST", body: JSON.stringify({ email, password }) });
    } catch (err) {
      if (err.message.includes("Invalid email or password") || err.message.includes("HTTP 401")) {
        const name = email.split("@")[0].replace(/[^a-zA-Z ]/g, " ").trim() || "User";
        await apiFetch("/auth/register", { method: "POST", body: JSON.stringify({ name, email, password }) });
        data = await apiFetch("/auth/login", { method: "POST", body: JSON.stringify({ email, password }) });
      } else { throw err; }
    }
    setToken(data.access_token);
    _userName = data.user?.name || email;
    const backendRole = data.user?.role || "USER";
    const mappedRole = backendRole === "ADMIN" ? "admin" : "user";
    if (role === "admin" && mappedRole !== "admin") { clearToken(); throw new Error("This account does not have admin access."); }
    return { token: data.access_token, user: { id: data.user?.id, name: data.user?.name, email, role: mappedRole } };
  },

  submitClaim: async (formData) => {
    const res = await fetch(`${BASE}/claims`, { method: "POST", headers: AUTH_TOKEN_HEADER(), body: formData });
    const json = await res.json().catch(() => ({}));
    if (!res.ok) {
      const details = json?.error?.details;
      const detailMsg = details ? Object.entries(details).map(([f, msgs]) => `${f}: ${[].concat(msgs).join(", ")}`).join("; ") : null;
      throw new Error(detailMsg || json?.error?.message || `HTTP ${res.status}`);
    }
    const claim = json.data?.claim;
    return { claimId: claim?.claim_reference || claim?.id, status: "Pending" };
  },

  getUserClaims: async () => {
    const data = await apiFetch("/claims");
    return (data.claims || []).map((c) => mapClaim(c, _userName));
  },

  getPendingClaims: async () => {
    const data = await apiFetch("/admin/claims?status=SUBMITTED&per_page=100");
    const claims = data.claims || [];
    let analyzed = [];
    try { const d2 = await apiFetch("/admin/claims?status=ANALYZED&per_page=100"); analyzed = d2.claims || []; } catch (_) {}
    const all = [...claims, ...analyzed];
    const userCache = {};
    const withNames = await Promise.all(all.map(async (c) => {
      if (!userCache[c.user_id]) {
        try { const ud = await apiFetch("/admin/users?per_page=100"); (ud.users || []).forEach(u => { userCache[u.id] = u.name; }); } catch (_) {}
      }
      return mapClaim(c, userCache[c.user_id] || "User");
    }));
    return withNames;
  },

  getClaimDetail: async (claimId) => {
    try { const data = await apiFetch(`/claims/${claimId}`); return mapClaim(data.claim || data, _userName); }
    catch (_) { const data = await apiFetch(`/admin/claims/${claimId}`); return mapClaim(data.claim || data, _userName); }
  },

  runFraudDetection: async (claimId) => {
    const data = await apiFetch(`/admin/claims/${claimId}/analyze`, { method: "POST", body: "{}" });
    const p = data.prediction || {};
    const isFraud = p.prediction === "Fraud";
    const accuracy = +(Math.max(p.fraud_probability ?? 0, p.non_fraud_probability ?? 0) * 100).toFixed(1);
    const approvalPct = Math.round((p.non_fraud_probability ?? 0) * 100);
    return { isFraud, accuracy, approvalPct, riskLevel: p.risk_level, recommendation: p.recommendation };
  },

  updateClaimStatus: async (claimId, status) => {
    const statusMap = { Approved: "APPROVED", Rejected: "REJECTED", Escalated: "ESCALATED", "Under Review": "UNDER_REVIEW" };
    const backendStatus = statusMap[status] || status.toUpperCase();
    await apiFetch(`/admin/claims/${claimId}/status`, { method: "PATCH", body: JSON.stringify({ status: backendStatus }) });
    return { success: true };
  },

  getAnalytics: async () => {
    const data = await apiFetch("/admin/statistics");
    const c = data.claims || {};
    const u = data.users || {};
    const total = c.total || 0;
    const fraudCount = c.by_prediction?.Fraud || 0;
    const genuineCount = c.by_prediction?.["Non-Fraud"] || 0;
    const by_risk = c.by_risk || {};
    const by_status = c.by_status || {};
    return {
      totalClaims: total, totalUsers: u.total || 0,
      fraudCount, genuineCount,
      pendingCount: (by_status.SUBMITTED || 0) + (by_status.ANALYZED || 0) + (by_status.UNDER_REVIEW || 0),
      approvedCount: by_status.APPROVED || 0,
      rejectedCount: by_status.REJECTED || 0,
      highRisk: by_risk.HIGH || 0, mediumRisk: by_risk.MEDIUM || 0, lowRisk: by_risk.LOW || 0,
      riskData: [
        { name: "High", value: by_risk.HIGH || 0, color: "#DC2626" },
        { name: "Medium", value: by_risk.MEDIUM || 0, color: "#D97706" },
        { name: "Low", value: by_risk.LOW || 0, color: "#16A34A" },
      ],
      predictionData: [
        { name: "Fraud", value: fraudCount, color: "#DC2626" },
        { name: "Genuine", value: genuineCount, color: "#16A34A" },
      ],
    };
  },
};

/* ─── Design tokens ──────────────────────────────────────────── */
const C = {
  bg: "#EEF2F7",
  surface: "#FFFFFF",
  border: "#D6DCE4",
  borderLight: "#E8ECF2",
  text: "#0D1B2A",
  textSec: "#4A5568",
  textMuted: "#9AA5B4",
  navy: "#0D1B2A",
  navyMed: "#1B3A5C",
  navyLight: "#2D5280",
  green: "#16A34A",
  greenBg: "#F0FDF4",
  greenDeep: "#14532D",
  greenBorder: "#86EFAC",
  amber: "#D97706",
  amberBg: "#FFFBEB",
  amberBorder: "#FCD34D",
  red: "#DC2626",
  redBg: "#FEF2F2",
  redDeep: "#7F1D1D",
  redBorder: "#FCA5A5",
  blue: "#2563EB",
  blueBg: "#EFF6FF",
  blueBorder: "#93C5FD",
  teal: "#0E7490",
  sky: "#38BDF8",
};

const inputStyle = {
  width: "100%", height: 38, padding: "0 12px", boxSizing: "border-box",
  border: `1px solid ${C.border}`, borderRadius: 6, fontSize: 13.5,
  fontFamily: "Inter, sans-serif", color: C.text, background: C.surface,
  outline: "none", transition: "border-color 0.15s, box-shadow 0.15s",
};

/* ─── Toast ──────────────────────────────────────────────────── */
const ToastCtx = createContext(null);
function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);
  const push = useCallback((msg, type = "success") => {
    const id = Date.now();
    setToasts(t => [...t, { id, msg, type }]);
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 4500);
  }, []);
  return (
    <ToastCtx.Provider value={push}>
      {children}
      <div style={{ position: "fixed", bottom: 24, right: 24, display: "flex", flexDirection: "column", gap: 8, zIndex: 9999 }}>
        {toasts.map(t => (
          <div key={t.id} style={{
            display: "flex", alignItems: "center", gap: 10,
            background: t.type === "error" ? "#FEF2F2" : t.type === "warning" ? "#FFFBEB" : "#F0FDF4",
            border: `1px solid ${t.type === "error" ? "#FCA5A5" : t.type === "warning" ? "#FCD34D" : "#86EFAC"}`,
            color: t.type === "error" ? "#991B1B" : t.type === "warning" ? "#92400E" : "#14532D",
            padding: "11px 16px", borderRadius: 8, fontSize: 13.5,
            fontFamily: "Inter, sans-serif", maxWidth: 380,
            boxShadow: "0 4px 16px rgba(13,27,42,0.12)",
          }}>
            {t.type === "error" ? <AlertTriangle size={15} /> : t.type === "warning" ? <AlertTriangle size={15} /> : <CheckCircle2 size={15} />}
            {t.msg}
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  );
}
const useToast = () => useContext(ToastCtx);

/* ─── Base UI ────────────────────────────────────────────────── */
function Spinner({ size = 16, color = "#FFF" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" style={{ animation: "spin 0.75s linear infinite", flexShrink: 0 }}>
      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
      <circle cx="12" cy="12" r="9" stroke={color} strokeWidth="2.5" strokeDasharray="36 20" strokeLinecap="round" />
    </svg>
  );
}

function Btn({ children, onClick, variant = "primary", size = "md", disabled, loading, type = "button", style: ext }) {
  const sizes = { sm: { padding: "5px 12px", fontSize: 12.5, gap: 5 }, md: { padding: "8px 16px", fontSize: 13.5, gap: 7 }, lg: { padding: "11px 22px", fontSize: 14.5, gap: 8 } };
  const variants = {
    primary:   { background: C.navy, color: "#FFF", border: "none" },
    secondary: { background: C.surface, color: C.text, border: `1px solid ${C.border}` },
    ghost:     { background: "transparent", color: C.textSec, border: "1px solid transparent" },
    danger:    { background: C.redBg, color: C.red, border: `1px solid ${C.redBorder}` },
    success:   { background: C.greenBg, color: C.greenDeep, border: `1px solid ${C.greenBorder}` },
    redsolid:  { background: C.redDeep, color: "#FFF", border: "none" },
    greensolid:{ background: C.greenDeep, color: "#FFF", border: "none" },
  };
  const s = sizes[size]; const v = variants[variant];
  return (
    <button type={type} onClick={onClick} disabled={disabled || loading}
      style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", gap: s.gap, padding: s.padding, fontSize: s.fontSize, fontWeight: 600, fontFamily: "Inter, sans-serif", borderRadius: 6, cursor: (disabled || loading) ? "not-allowed" : "pointer", opacity: disabled ? 0.5 : 1, transition: "background 0.15s, opacity 0.15s", ...v, ...ext }}>
      {loading && <Spinner size={13} color={variant === "primary" || variant === "redsolid" || variant === "greensolid" ? "#FFF" : C.navy} />}
      {children}
    </button>
  );
}

function Badge({ children, color, bg, border }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 4, padding: "2px 9px", borderRadius: 12, fontSize: 11.5, fontWeight: 700, fontFamily: "Inter, sans-serif", color: color || C.textSec, background: bg || C.borderLight, border: `1px solid ${border || C.border}`, whiteSpace: "nowrap", letterSpacing: "0.03em" }}>
      {children}
    </span>
  );
}

function StatusBadge({ status }) {
  const m = {
    Pending:      { color: C.amber,  bg: C.amberBg,  border: C.amberBorder },
    "Under Review":{ color: C.blue,   bg: C.blueBg,   border: C.blueBorder },
    Approved:     { color: C.greenDeep, bg: C.greenBg, border: C.greenBorder },
    Rejected:     { color: C.redDeep, bg: C.redBg,    border: C.redBorder },
    Escalated:    { color: C.redDeep, bg: C.redBg,    border: C.redBorder },
  };
  const s = m[status] || { color: C.textSec, bg: C.borderLight, border: C.border };
  return <Badge color={s.color} bg={s.bg} border={s.border}>{status}</Badge>;
}

function RiskBadge({ risk }) {
  if (!risk) return <span style={{ color: C.textMuted, fontSize: 12 }}>—</span>;
  const m = { HIGH: { color: C.redDeep, bg: C.redBg, border: C.redBorder }, MEDIUM: { color: C.amber, bg: C.amberBg, border: C.amberBorder }, LOW: { color: C.greenDeep, bg: C.greenBg, border: C.greenBorder } };
  const s = m[risk.toUpperCase()] || { color: C.textSec, bg: C.borderLight, border: C.border };
  return <Badge color={s.color} bg={s.bg} border={s.border}>{risk}</Badge>;
}

function StatCard({ label, value, icon: Icon, color, accent }) {
  const ac = accent || color || C.navy;
  return (
    <div style={{ background: C.surface, borderRadius: 10, padding: "20px 22px", borderLeft: `4px solid ${ac}`, boxShadow: "0 1px 4px rgba(13,27,42,0.07), 0 0 0 1px rgba(13,27,42,0.04)" }}>
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
        <div>
          <p style={{ fontFamily: "Inter, sans-serif", fontSize: 11, fontWeight: 700, color: C.textMuted, margin: "0 0 10px", textTransform: "uppercase", letterSpacing: "0.09em" }}>{label}</p>
          <p style={{ fontFamily: "Inter, sans-serif", fontSize: 32, fontWeight: 800, color: color || C.text, margin: 0, lineHeight: 1 }}>{value ?? "—"}</p>
        </div>
        {Icon && (
          <div style={{ width: 40, height: 40, borderRadius: 10, background: `${ac}18`, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <Icon size={18} color={ac} />
          </div>
        )}
      </div>
    </div>
  );
}

function Modal({ open, onClose, title, width = 700, children, footer }) {
  if (!open) return null;
  return (
    <div onClick={onClose} style={{ position: "fixed", inset: 0, background: "rgba(13,27,42,0.5)", zIndex: 1000, display: "flex", alignItems: "center", justifyContent: "center", padding: 20 }}>
      <div onClick={e => e.stopPropagation()} style={{ background: C.surface, borderRadius: 12, width: "100%", maxWidth: width, maxHeight: "92vh", display: "flex", flexDirection: "column", boxShadow: "0 24px 64px rgba(13,27,42,0.2)", border: `1px solid ${C.border}` }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "18px 24px", borderBottom: `1px solid ${C.border}`, flexShrink: 0 }}>
          <h3 style={{ fontFamily: "Inter, sans-serif", fontWeight: 700, fontSize: 16, color: C.text, margin: 0 }}>{title}</h3>
          <button onClick={onClose} style={{ background: C.borderLight, border: "none", cursor: "pointer", color: C.textMuted, width: 30, height: 30, borderRadius: 6, display: "flex", alignItems: "center", justifyContent: "center" }}><X size={15} /></button>
        </div>
        <div style={{ overflowY: "auto", flex: 1, padding: 24 }}>{children}</div>
        {footer && <div style={{ padding: "14px 24px", borderTop: `1px solid ${C.border}`, flexShrink: 0, display: "flex", gap: 8, justifyContent: "flex-end", background: "#FAFBFC", borderRadius: "0 0 12px 12px" }}>{footer}</div>}
      </div>
    </div>
  );
}

function ConfirmDialog({ open, onClose, onConfirm, title, message, confirmLabel = "Confirm", loading }) {
  return (
    <Modal open={open} onClose={onClose} title={title} width={420}
      footer={<><Btn variant="secondary" onClick={onClose} disabled={loading}>Cancel</Btn><Btn variant="danger" onClick={onConfirm} loading={loading}>{confirmLabel}</Btn></>}>
      <p style={{ fontFamily: "Inter, sans-serif", fontSize: 14, color: C.textSec, margin: 0, lineHeight: 1.7 }}>{message}</p>
    </Modal>
  );
}

function EmptyState({ icon: Icon, title, subtitle }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "48px 24px", textAlign: "center" }}>
      <div style={{ width: 48, height: 48, borderRadius: 12, background: C.borderLight, display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 14 }}>
        {Icon && <Icon size={22} color={C.textMuted} />}
      </div>
      <p style={{ fontFamily: "Inter, sans-serif", fontWeight: 700, fontSize: 15, color: C.text, margin: "0 0 6px" }}>{title}</p>
      {subtitle && <p style={{ fontFamily: "Inter, sans-serif", fontSize: 13, color: C.textSec, margin: 0, maxWidth: 320, lineHeight: 1.6 }}>{subtitle}</p>}
    </div>
  );
}

/* ─── Sidebar ────────────────────────────────────────────────── */
function Sidebar({ items, active, onSelect, userName, role, onLogout, mobileOpen, onClose }) {
  const sidebarBg = "linear-gradient(180deg, #0D1B2A 0%, #162535 100%)";
  const content = (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", padding: "0" }}>
      {/* Brand header */}
      <div style={{ padding: "20px 18px 16px", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 11 }}>
          <div style={{ width: 34, height: 34, borderRadius: 8, background: C.teal, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
            <Shield size={17} color="#FFF" />
          </div>
          <div>
            <div style={{ fontFamily: "Inter, sans-serif", fontWeight: 800, fontSize: 14, color: "#FFFFFF", letterSpacing: "0.04em" }}>CLAIM VISION</div>
            <div style={{ fontFamily: "Inter, sans-serif", fontSize: 10.5, color: C.sky, opacity: 0.7, letterSpacing: "0.08em" }}>INSURANCE · IMS</div>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav style={{ flex: 1, padding: "12px 10px", display: "flex", flexDirection: "column", gap: 2 }}>
        {items.map(item => {
          const isActive = active === item.key;
          return (
            <button key={item.key} onClick={() => { onSelect(item.key); onClose?.(); }}
              style={{ display: "flex", alignItems: "center", gap: 10, width: "100%", padding: "9px 12px", borderRadius: 7, border: isActive ? `1px solid rgba(56,189,248,0.2)` : "1px solid transparent", cursor: "pointer", textAlign: "left", fontFamily: "Inter, sans-serif", fontSize: 13.5, fontWeight: isActive ? 700 : 400, color: isActive ? "#FFFFFF" : "#94A3B8", background: isActive ? "rgba(255,255,255,0.1)" : "transparent", transition: "all 0.15s", borderLeft: isActive ? `3px solid ${C.sky}` : "3px solid transparent", paddingLeft: isActive ? 10 : 12 }}
              onMouseEnter={e => { if (!isActive) { e.currentTarget.style.background = "rgba(255,255,255,0.06)"; e.currentTarget.style.color = "#CBD5E1"; } }}
              onMouseLeave={e => { if (!isActive) { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "#94A3B8"; } }}>
              <item.icon size={15} />
              {item.label}
            </button>
          );
        })}
      </nav>

      {/* User + logout */}
      <div style={{ padding: "12px 10px 16px", borderTop: "1px solid rgba(255,255,255,0.06)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 12px", borderRadius: 8, background: "rgba(255,255,255,0.04)", marginBottom: 6 }}>
          <div style={{ width: 30, height: 30, borderRadius: "50%", background: C.navyLight, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12.5, fontWeight: 800, color: "#FFF", fontFamily: "Inter, sans-serif", flexShrink: 0, border: "1px solid rgba(255,255,255,0.1)" }}>
            {(userName || "U")[0].toUpperCase()}
          </div>
          <div style={{ minWidth: 0 }}>
            <div style={{ fontFamily: "Inter, sans-serif", fontSize: 13, fontWeight: 600, color: "#E2E8F0", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{userName || "User"}</div>
            <div style={{ fontFamily: "Inter, sans-serif", fontSize: 11, color: "#64748B" }}>{role === "admin" ? "Administrator" : "User"}</div>
          </div>
        </div>
        <button onClick={onLogout}
          style={{ display: "flex", alignItems: "center", gap: 8, width: "100%", padding: "8px 12px", borderRadius: 7, border: "none", cursor: "pointer", background: "transparent", color: "#64748B", fontFamily: "Inter, sans-serif", fontSize: 13, fontWeight: 500, transition: "all 0.15s" }}
          onMouseEnter={e => { e.currentTarget.style.background = "rgba(239,68,68,0.12)"; e.currentTarget.style.color = "#FCA5A5"; }}
          onMouseLeave={e => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "#64748B"; }}>
          <LogOut size={14} /> Sign out
        </button>
      </div>
    </div>
  );

  return (
    <>
      <div className="meridian-sidebar" style={{ width: 230, flexShrink: 0, background: sidebarBg, height: "100vh", overflowY: "auto", position: "sticky", top: 0 }}>
        {content}
      </div>
      {mobileOpen && (
        <div style={{ position: "fixed", inset: 0, zIndex: 500 }}>
          <div onClick={onClose} style={{ position: "absolute", inset: 0, background: "rgba(0,0,0,0.5)" }} />
          <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: 230, background: sidebarBg }}>
            {content}
          </div>
        </div>
      )}
      <style>{`.meridian-sidebar{display:flex;flex-direction:column}@media(max-width:768px){.meridian-sidebar{display:none!important}}`}</style>
    </>
  );
}

function TopBar({ title, subtitle, actions, onMenu }) {
  return (
    <div style={{ background: C.surface, borderBottom: `1px solid ${C.border}`, padding: "0 28px", height: 60, display: "flex", alignItems: "center", justifyContent: "space-between", flexShrink: 0, boxShadow: "0 1px 4px rgba(13,27,42,0.05)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
        <button onClick={onMenu} className="menu-toggle" style={{ display: "none", background: "none", border: "none", cursor: "pointer", color: C.textSec, padding: 4 }}><Menu size={20} /></button>
        <div>
          <h1 style={{ fontFamily: "Inter, sans-serif", fontWeight: 800, fontSize: 17, color: C.text, margin: 0 }}>{title}</h1>
          {subtitle && <p style={{ fontFamily: "Inter, sans-serif", fontSize: 12, color: C.textMuted, margin: 0 }}>{subtitle}</p>}
        </div>
      </div>
      {actions && <div style={{ display: "flex", gap: 8 }}>{actions}</div>}
      <style>{`@media(max-width:768px){.menu-toggle{display:flex!important}}`}</style>
    </div>
  );
}

/* ─── Claim thumbnail ────────────────────────────────────────── */
function ClaimThumb({ imageUrl, size = 44 }) {
  return (
    <div style={{ width: size, height: size, borderRadius: 8, overflow: "hidden", background: C.borderLight, flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center", border: `1px solid ${C.border}` }}>
      {imageUrl ? <img src={imageUrl} alt="" style={{ width: "100%", height: "100%", objectFit: "cover" }} /> : <Car size={16} color={C.textMuted} />}
    </div>
  );
}

/* ─── Fraud flag card (admin review) ─────────────────────────── */
function FraudClaimCard({ claim, onReview, tone }) {
  const isFraud = tone === "fraud";
  const pct = claim.accuracy ?? 0;
  const barColor = isFraud ? C.red : C.green;
  const cardBg = isFraud ? "#FFF8F8" : "#F6FFF8";
  const [hovered, setHovered] = useState(false);

  return (
    <div
      onClick={() => onReview(claim.id)}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{ display: "flex", alignItems: "center", gap: 14, padding: "14px 18px", cursor: "pointer", background: hovered ? (isFraud ? "#FFF0F0" : "#EDFBF0") : cardBg, borderBottom: `1px solid ${isFraud ? "#FCE4E4" : "#D4F4DE"}`, transition: "background 0.15s" }}>
      <ClaimThumb imageUrl={claim.imageUrl} size={48} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 12.5, fontWeight: 700, color: C.text, marginBottom: 3, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{claim.ref}</div>
        <div style={{ fontFamily: "Inter, sans-serif", fontSize: 12, color: C.textSec, marginBottom: 8 }}>
          {claim.policyNumber || "—"} · ₹{Number(claim.estimatedCost || 0).toLocaleString("en-IN")}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ flex: 1, height: 5, background: isFraud ? "#FECACA" : "#BBF7D0", borderRadius: 3, overflow: "hidden" }}>
            <div style={{ height: "100%", width: `${pct}%`, background: barColor, borderRadius: 3 }} />
          </div>
          <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 12, fontWeight: 700, color: barColor, minWidth: 36 }}>{pct}%</span>
        </div>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 6, color: C.textMuted }}>
        <Eye size={14} />
      </div>
    </div>
  );
}

/* ─── Flag column (admin) ────────────────────────────────────── */
function FlagColumn({ tone, title, subtitle, claims, onReview, onApproveAll, approvingAll }) {
  const isFraud = tone === "fraud";
  const headerBg = isFraud ? C.redDeep : C.greenDeep;
  const Icon = isFraud ? ShieldAlert : ShieldCheck;
  return (
    <div style={{ borderRadius: 10, overflow: "hidden", boxShadow: "0 2px 8px rgba(13,27,42,0.08)", border: `1px solid ${isFraud ? C.redBorder : C.greenBorder}` }}>
      {/* Header */}
      <div style={{ background: headerBg, padding: "14px 18px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 34, height: 34, borderRadius: 8, background: "rgba(255,255,255,0.15)", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <Icon size={17} color="#FFF" />
          </div>
          <div>
            <div style={{ fontFamily: "Inter, sans-serif", fontWeight: 800, fontSize: 14, color: "#FFF" }}>{title}</div>
            <div style={{ fontFamily: "Inter, sans-serif", fontSize: 11.5, color: "rgba(255,255,255,0.65)" }}>{subtitle} · <strong style={{ color: "#FFF" }}>{claims.length}</strong> claim{claims.length !== 1 ? "s" : ""}</div>
          </div>
        </div>
        {!isFraud && onApproveAll && claims.length > 0 && (
          <Btn variant="greensolid" size="sm" onClick={e => { e.stopPropagation(); onApproveAll(); }} loading={approvingAll}
            style={{ background: "rgba(255,255,255,0.2)", color: "#FFF", border: "1px solid rgba(255,255,255,0.3)" }}>
            <CheckCircle2 size={12} /> Approve all
          </Btn>
        )}
      </div>
      {/* Body */}
      <div style={{ background: C.surface }}>
        {claims.length === 0 ? (
          <EmptyState icon={isFraud ? ShieldAlert : ShieldCheck} title="Nothing here" subtitle={`No ${isFraud ? "flagged" : "genuine"} claims pending review.`} />
        ) : (
          claims.map((c, i) => <FraudClaimCard key={c.id} claim={c} onReview={onReview} tone={tone} />)
        )}
      </div>
    </div>
  );
}

/* ─── Claim detail modal ─────────────────────────────────────── */
function ClaimDetailModal({ claimId, onClose, onStatusChanged, isAdmin }) {
  const push = useToast();
  const [claim, setClaim] = useState(null);
  const [loading, setLoading] = useState(true);
  const [detecting, setDetecting] = useState(false);
  const [detectResult, setDetectResult] = useState(null);
  const [actionLoading, setActionLoading] = useState("");
  const [confirmAction, setConfirmAction] = useState(null);
  const [imageZoom, setImageZoom] = useState(false);

  useEffect(() => {
    let alive = true;
    api.getClaimDetail(claimId)
      .then(c => { if (alive) { setClaim(c); if (c.isFraud !== null) setDetectResult({ isFraud: c.isFraud, accuracy: c.accuracy, approvalPct: c.approvalPct, riskLevel: c.riskLevel, recommendation: c.recommendation }); } })
      .catch(() => { if (alive) push("Failed to load claim.", "error"); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, [claimId]);

  const runDetection = async () => {
    setDetecting(true); setDetectResult(null);
    try { setDetectResult(await api.runFraudDetection(claimId)); }
    catch { push("Detection service unavailable.", "error"); }
    finally { setDetecting(false); }
  };

  useEffect(() => {
    if (isAdmin && claim && claim.isFraud === null && !detectResult && !detecting) runDetection();
  }, [isAdmin, claim]);

  const act = async (status) => {
    setActionLoading(status); setConfirmAction(null);
    try { await api.updateClaimStatus(claimId, status); push(`Claim marked as ${status}.`); onStatusChanged?.(claimId, status); onClose(); }
    catch { push("Failed to update status.", "error"); }
    finally { setActionLoading(""); }
  };

  const footer = isAdmin && !loading && claim ? (
    <>
      <Btn variant="secondary" onClick={onClose}>Close</Btn>
      <Btn variant="success" onClick={() => setConfirmAction("Approved")} loading={actionLoading === "Approved"} disabled={!!actionLoading}><CheckCircle2 size={13} /> Approve</Btn>
      <Btn variant="ghost" onClick={() => act("Under Review")} loading={actionLoading === "Under Review"} disabled={!!actionLoading}><Clock size={13} /> Under Review</Btn>
      <Btn variant="danger" onClick={() => setConfirmAction("Rejected")} loading={actionLoading === "Rejected"} disabled={!!actionLoading}><XCircle size={13} /> Reject</Btn>
      <Btn variant="danger" onClick={() => setConfirmAction("Escalated")} loading={actionLoading === "Escalated"} disabled={!!actionLoading}><AlertOctagon size={13} /> Escalate</Btn>
    </>
  ) : <Btn variant="secondary" onClick={onClose}>Close</Btn>;

  return (
    <>
      <Modal open onClose={onClose} title={loading ? "Loading…" : `Claim — ${claim?.ref}`} width={840} footer={footer}>
        {loading ? (
          <div style={{ display: "flex", justifyContent: "center", padding: 48 }}><Spinner size={26} color={C.navy} /></div>
        ) : claim ? (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }} className="detail-grid">
            {/* Left */}
            <div style={{ display: "flex", flexDirection: "column", gap: 22 }}>
              <Section title="Claimant & Policy">
                <Row label="Claimant" value={claim.claimantName} />
                <Row label="Policy / Vehicle No." value={claim.policyNumber} mono />
                <Row label="Vehicle" value={[claim.vehicleModel, claim.vehicleYear].filter(Boolean).join(" · ") || "—"} />
              </Section>
              <Section title="Incident">
                <Row label="Date" value={claim.incidentDate} />
                <Row label="Claim Amount" value={claim.estimatedCost ? `₹${Number(claim.estimatedCost).toLocaleString("en-IN")}` : "—"} mono />
                <Row label="Status" value={<StatusBadge status={claim.status} />} />
              </Section>
              <Section title="Description">
                <p style={{ fontFamily: "Inter, sans-serif", fontSize: 13.5, color: C.textSec, margin: 0, lineHeight: 1.65 }}>{claim.description || "—"}</p>
              </Section>
            </div>
            {/* Right */}
            <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
              <Section title="Damage Image">
                {claim.imageUrl ? (
                  <div style={{ borderRadius: 8, overflow: "hidden", border: `1px solid ${C.border}`, cursor: "zoom-in", position: "relative" }} onClick={() => setImageZoom(true)}>
                    <img src={claim.imageUrl} alt="Damage" style={{ width: "100%", height: 210, objectFit: "cover", display: "block" }} />
                    <div style={{ position: "absolute", bottom: 8, right: 8, background: "rgba(13,27,42,0.6)", color: "#FFF", fontSize: 11, padding: "3px 8px", borderRadius: 4, fontFamily: "Inter, sans-serif" }}>Click to enlarge</div>
                  </div>
                ) : (
                  <div style={{ height: 160, border: `1.5px dashed ${C.border}`, borderRadius: 8, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 8, background: C.bg }}>
                    <Car size={30} color={C.textMuted} />
                    <span style={{ fontFamily: "Inter, sans-serif", fontSize: 12.5, color: C.textMuted }}>No image uploaded</span>
                  </div>
                )}
              </Section>
              {isAdmin && (
                <Section title="AI Fraud Analysis">
                  <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 12 }}>
                    <Btn variant="ghost" size="sm" onClick={runDetection} disabled={detecting}><RefreshCw size={11} /> Re-analyze</Btn>
                  </div>
                  {detecting ? (
                    <div style={{ display: "flex", alignItems: "center", gap: 10, color: C.textSec, fontFamily: "Inter, sans-serif", fontSize: 13.5, padding: "12px 0" }}>
                      <Spinner size={16} color={C.navy} /> Analyzing image…
                    </div>
                  ) : detectResult ? (
                    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                        <div style={{ padding: "12px 14px", borderRadius: 8, background: detectResult.isFraud ? C.redBg : C.greenBg, border: `1px solid ${detectResult.isFraud ? C.redBorder : C.greenBorder}` }}>
                          <div style={{ fontFamily: "Inter, sans-serif", fontSize: 10.5, fontWeight: 700, color: detectResult.isFraud ? C.redDeep : C.greenDeep, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 4 }}>Prediction</div>
                          <div style={{ fontFamily: "Inter, sans-serif", fontSize: 15, fontWeight: 800, color: detectResult.isFraud ? C.red : C.green }}>{detectResult.isFraud ? "Potential Fraud" : "Likely Genuine"}</div>
                        </div>
                        <div style={{ padding: "12px 14px", borderRadius: 8, background: C.bg, border: `1px solid ${C.border}` }}>
                          <div style={{ fontFamily: "Inter, sans-serif", fontSize: 10.5, fontWeight: 700, color: C.textMuted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 4 }}>Risk Level</div>
                          <RiskBadge risk={detectResult.riskLevel} />
                        </div>
                      </div>
                      <div>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                          <span style={{ fontFamily: "Inter, sans-serif", fontSize: 12.5, color: C.textSec }}>Fraud probability</span>
                          <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 13, fontWeight: 700, color: detectResult.isFraud ? C.red : C.green }}>{detectResult.accuracy ?? 0}%</span>
                        </div>
                        <div style={{ height: 8, background: C.borderLight, borderRadius: 4, overflow: "hidden" }}>
                          <div style={{ height: "100%", width: `${detectResult.accuracy ?? 0}%`, background: (detectResult.accuracy ?? 0) > 70 ? C.red : (detectResult.accuracy ?? 0) > 40 ? C.amber : C.green, borderRadius: 4, transition: "width 0.6s ease" }} />
                        </div>
                      </div>
                      {detectResult.recommendation && (
                        <div style={{ display: "flex", gap: 8, padding: "10px 12px", background: C.blueBg, borderRadius: 7, border: `1px solid ${C.blueBorder}` }}>
                          <Info size={14} color={C.blue} style={{ flexShrink: 0, marginTop: 1 }} />
                          <p style={{ fontFamily: "Inter, sans-serif", fontSize: 12.5, color: C.textSec, margin: 0, lineHeight: 1.55 }}>{detectResult.recommendation}</p>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div>
                      <p style={{ fontFamily: "Inter, sans-serif", fontSize: 13, color: C.textMuted, margin: "0 0 10px" }}>No analysis available.</p>
                      <Btn variant="primary" size="sm" onClick={runDetection}>Run Analysis</Btn>
                    </div>
                  )}
                </Section>
              )}
            </div>
          </div>
        ) : null}
        <style>{`@media(max-width:640px){.detail-grid{grid-template-columns:1fr!important}}`}</style>
      </Modal>

      {confirmAction && (
        <ConfirmDialog open title={`${confirmAction} this claim?`}
          message={`Are you sure you want to mark claim ${claim?.ref} as ${confirmAction.toLowerCase()}? This will update the claim status immediately.`}
          confirmLabel={confirmAction} loading={!!actionLoading}
          onClose={() => setConfirmAction(null)} onConfirm={() => act(confirmAction)} />
      )}

      {imageZoom && claim?.imageUrl && (
        <div onClick={() => setImageZoom(false)} style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.88)", zIndex: 2000, display: "flex", alignItems: "center", justifyContent: "center", cursor: "zoom-out" }}>
          <img src={claim.imageUrl} alt="Damage" style={{ maxWidth: "90vw", maxHeight: "90vh", objectFit: "contain", borderRadius: 8 }} />
        </div>
      )}
    </>
  );
}

function Section({ title, children }) {
  return (
    <div>
      <p style={{ fontFamily: "Inter, sans-serif", fontSize: 10.5, fontWeight: 800, color: C.textMuted, margin: "0 0 12px", textTransform: "uppercase", letterSpacing: "0.1em" }}>{title}</p>
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>{children}</div>
    </div>
  );
}

function Row({ label, value, mono }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", gap: 16, alignItems: "flex-start" }}>
      <span style={{ fontFamily: "Inter, sans-serif", fontSize: 13, color: C.textMuted, flexShrink: 0 }}>{label}</span>
      <span style={{ fontFamily: mono ? "'IBM Plex Mono', monospace" : "Inter, sans-serif", fontSize: 13, color: C.text, textAlign: "right", fontWeight: mono ? 600 : 400 }}>{value ?? "—"}</span>
    </div>
  );
}

/* ─── Claims table ───────────────────────────────────────────── */
function ClaimsTable({ claims, loading, onRowClick, emptyTitle = "No claims", emptySubtitle }) {
  const cols = [
    { label: "", key: "img", w: 56 },
    { label: "Claim Ref", key: "ref" },
    { label: "Policy / Vehicle", key: "policy" },
    { label: "Amount", key: "amount" },
    { label: "Date", key: "date" },
    { label: "Status", key: "status" },
    { label: "", key: "action" },
  ];
  return (
    <div style={{ background: C.surface, borderRadius: 10, border: `1px solid ${C.border}`, overflow: "hidden", boxShadow: "0 1px 4px rgba(13,27,42,0.06)" }}>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "Inter, sans-serif" }}>
          <thead>
            <tr style={{ background: C.bg }}>
              {cols.map(c => (
                <th key={c.key} style={{ padding: "10px 16px", fontSize: 11, fontWeight: 700, color: C.textMuted, textAlign: "left", borderBottom: `1px solid ${C.border}`, textTransform: "uppercase", letterSpacing: "0.07em", whiteSpace: "nowrap" }}>
                  {c.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading ? (
              Array.from({ length: 5 }).map((_, i) => (
                <tr key={i}>
                  {cols.map(c => (
                    <td key={c.key} style={{ padding: "12px 16px", borderBottom: `1px solid ${C.borderLight}` }}>
                      <div style={{ height: 14, borderRadius: 4, background: C.borderLight, width: c.key === "img" ? 36 : "70%" }} />
                    </td>
                  ))}
                </tr>
              ))
            ) : claims.length === 0 ? (
              <tr><td colSpan={cols.length}><EmptyState title={emptyTitle} subtitle={emptySubtitle} /></td></tr>
            ) : (
              claims.map((r, i) => (
                <tr key={r.id} onClick={() => onRowClick?.(r.id)}
                  style={{ borderBottom: i < claims.length - 1 ? `1px solid ${C.borderLight}` : "none", cursor: onRowClick ? "pointer" : "default", transition: "background 0.1s" }}
                  onMouseEnter={e => { if (onRowClick) e.currentTarget.style.background = C.bg; }}
                  onMouseLeave={e => { e.currentTarget.style.background = ""; }}>
                  <td style={{ padding: "10px 16px" }}><ClaimThumb imageUrl={r.imageUrl} size={38} /></td>
                  <td style={{ padding: "10px 16px", fontFamily: "'IBM Plex Mono', monospace", fontSize: 12.5, fontWeight: 600, color: C.text, whiteSpace: "nowrap" }}>{r.ref}</td>
                  <td style={{ padding: "10px 16px", fontSize: 13, color: C.textSec, whiteSpace: "nowrap" }}>{r.policyNumber || "—"}</td>
                  <td style={{ padding: "10px 16px", fontFamily: "'IBM Plex Mono', monospace", fontSize: 12.5, whiteSpace: "nowrap" }}>₹{Number(r.estimatedCost || 0).toLocaleString("en-IN")}</td>
                  <td style={{ padding: "10px 16px", fontSize: 13, color: C.textSec, whiteSpace: "nowrap" }}>{r.incidentDate || "—"}</td>
                  <td style={{ padding: "10px 16px" }}><StatusBadge status={r.status} /></td>
                  <td style={{ padding: "10px 16px" }}>
                    <Btn variant="ghost" size="sm" onClick={e => { e.stopPropagation(); onRowClick?.(r.id); }}><Eye size={13} /></Btn>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ─── Claim submit form ──────────────────────────────────────── */
function ClaimSubmitForm({ onSuccess, onCancel }) {
  const push = useToast();
  const fileRef = useRef(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [form, setForm] = useState({ policyNumber: "", incidentDate: "", estimatedCost: "", description: "" });
  const set = k => e => setForm(f => ({ ...f, [k]: e.target.value }));
  const onFile = f => { if (!f) return; setFile(f); setPreview(URL.createObjectURL(f)); };
  const canSubmit = file && form.policyNumber && form.incidentDate && form.estimatedCost && form.description;

  const submit = async () => {
    if (!canSubmit) { push("Fill all fields and upload an image.", "warning"); return; }
    setSubmitting(true);
    try {
      const fd = new FormData();
      fd.append("vehicle_image", file);
      fd.append("vehicle_number", form.policyNumber);
      fd.append("claim_amount", form.estimatedCost);
      fd.append("incident_date", form.incidentDate);
      fd.append("description", form.description);
      fd.append("claim_reference", `CLM-${Date.now()}`);
      fd.append("vehicle_model", "Unknown");
      fd.append("vehicle_year", new Date(form.incidentDate).getFullYear() || new Date().getFullYear());
      const res = await api.submitClaim(fd);
      push(`Claim ${res.claimId} submitted successfully.`);
      onSuccess?.(res);
    } catch (e) { push(e.message || "Failed to submit claim.", "error"); }
    finally { setSubmitting(false); }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div>
        <label style={{ display: "block", fontFamily: "Inter, sans-serif", fontSize: 11.5, fontWeight: 700, color: C.textSec, marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.06em" }}>Damage Photo <span style={{ color: C.red }}>*</span></label>
        {preview ? (
          <div style={{ position: "relative", borderRadius: 8, overflow: "hidden", border: `1px solid ${C.border}` }}>
            <img src={preview} alt="Preview" style={{ width: "100%", height: 190, objectFit: "cover", display: "block" }} />
            <button onClick={() => { setFile(null); setPreview(null); }} style={{ position: "absolute", top: 8, right: 8, background: "rgba(0,0,0,0.65)", border: "none", borderRadius: 6, color: "#FFF", cursor: "pointer", padding: "5px 7px", display: "flex", alignItems: "center" }}><X size={13} /></button>
            <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, background: "rgba(13,27,42,0.55)", color: "#FFF", fontSize: 12, padding: "5px 12px", fontFamily: "Inter, sans-serif" }}>{file?.name}</div>
          </div>
        ) : (
          <div onClick={() => fileRef.current.click()}
            style={{ border: `1.5px dashed ${C.border}`, borderRadius: 8, padding: "28px 20px", textAlign: "center", cursor: "pointer", background: C.bg, transition: "border-color 0.15s, background 0.15s" }}
            onMouseEnter={e => { e.currentTarget.style.borderColor = C.navy; e.currentTarget.style.background = "#E6EBF2"; }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = C.border; e.currentTarget.style.background = C.bg; }}>
            <Upload size={22} color={C.textMuted} style={{ marginBottom: 8 }} />
            <p style={{ fontFamily: "Inter, sans-serif", fontSize: 14, fontWeight: 700, color: C.text, margin: "0 0 4px" }}>Upload damage photo</p>
            <p style={{ fontFamily: "Inter, sans-serif", fontSize: 12.5, color: C.textMuted, margin: 0 }}>JPEG · PNG · WEBP · Max 10 MB</p>
          </div>
        )}
        <input ref={fileRef} type="file" accept="image/*" hidden onChange={e => onFile(e.target.files[0])} />
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }} className="form2col">
        <Field label="Policy / Vehicle No." required>
          <input value={form.policyNumber} onChange={set("policyNumber")} placeholder="e.g. POL-00000" style={inputStyle} />
        </Field>
        <Field label="Incident Date" required>
          <input type="date" value={form.incidentDate} onChange={set("incidentDate")} style={inputStyle} />
        </Field>
      </div>
      <Field label="Claim Amount (₹)" required>
        <input type="number" value={form.estimatedCost} onChange={set("estimatedCost")} placeholder="e.g. 25000" style={inputStyle} />
      </Field>
      <Field label="Description of Damage" required>
        <textarea value={form.description} onChange={set("description")} rows={3} placeholder="Describe what happened and where the damage occurred." style={{ ...inputStyle, height: "auto", padding: "10px 12px", resize: "vertical", lineHeight: 1.6 }} />
      </Field>
      <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
        {onCancel && <Btn variant="secondary" onClick={onCancel}>Cancel</Btn>}
        <Btn variant="primary" onClick={submit} loading={submitting} disabled={!canSubmit} size="md">Submit Claim</Btn>
      </div>
      <style>{`@media(max-width:480px){.form2col{grid-template-columns:1fr!important}}`}</style>
    </div>
  );
}

function Field({ label, required, children }) {
  return (
    <div>
      <label style={{ display: "block", fontFamily: "Inter, sans-serif", fontSize: 12, fontWeight: 600, color: C.textSec, marginBottom: 6 }}>
        {label}{required && <span style={{ color: C.red, marginLeft: 2 }}>*</span>}
      </label>
      {children}
    </div>
  );
}

/* ─── User App ───────────────────────────────────────────────── */
function UserApp({ user, onLogout }) {
  const push = useToast();
  const [tab, setTab] = useState("overview");
  const [claims, setClaims] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showSubmit, setShowSubmit] = useState(false);
  const [detailId, setDetailId] = useState(null);
  const [search, setSearch] = useState("");
  const [mobileOpen, setMobileOpen] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try { setClaims(await api.getUserClaims(user.id)); }
    catch { push("Failed to load claims.", "error"); }
    finally { setLoading(false); }
  }, [user.id]);

  useEffect(() => { load(); }, [load]);

  const total = claims.length;
  const pending = claims.filter(c => c.status === "Pending" || c.status === "Under Review").length;
  const approved = claims.filter(c => c.status === "Approved").length;
  const rejected = claims.filter(c => c.status === "Rejected").length;

  const filtered = claims.filter(c =>
    !search || c.ref.toLowerCase().includes(search.toLowerCase()) || (c.policyNumber || "").toLowerCase().includes(search.toLowerCase())
  );

  const navItems = [
    { key: "overview", label: "Overview", icon: LayoutDashboard },
    { key: "claims", label: "My Claims", icon: FileText },
  ];

  return (
    <div style={{ display: "flex", height: "100vh", overflow: "hidden", background: C.bg }}>
      <Sidebar items={navItems} active={tab} onSelect={setTab} userName={user.name} role="user" onLogout={onLogout} mobileOpen={mobileOpen} onClose={() => setMobileOpen(false)} />
      <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <TopBar
          title={tab === "overview" ? "Overview" : "My Claims"}
          onMenu={() => setMobileOpen(true)}
          actions={<Btn variant="primary" onClick={() => setShowSubmit(true)}><Upload size={14} /> New Claim</Btn>}
        />
        <div style={{ flex: 1, padding: 28, overflowY: "auto" }}>
          {tab === "overview" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 14 }} className="user-stats">
                <StatCard label="Total Claims" value={total} icon={FileText} accent={C.navy} />
                <StatCard label="Under Review" value={pending} icon={Clock} color={C.amber} />
                <StatCard label="Approved" value={approved} icon={CheckCircle2} color={C.green} />
                <StatCard label="Rejected" value={rejected} icon={XCircle} color={C.red} />
              </div>
              <div style={{ background: C.surface, borderRadius: 10, border: `1px solid ${C.border}`, overflow: "hidden", boxShadow: "0 1px 4px rgba(13,27,42,0.06)" }}>
                <div style={{ padding: "14px 20px", borderBottom: `1px solid ${C.border}`, display: "flex", alignItems: "center", justifyContent: "space-between", background: C.bg }}>
                  <h2 style={{ fontFamily: "Inter, sans-serif", fontWeight: 700, fontSize: 14, color: C.text, margin: 0 }}>Recent Claims</h2>
                  <Btn variant="ghost" size="sm" onClick={load}><RefreshCw size={13} /></Btn>
                </div>
                <ClaimsTable claims={filtered.slice(0, 8)} loading={loading} onRowClick={id => setDetailId(id)} emptyTitle="No claims yet" emptySubtitle="Submit your first claim to get started." />
              </div>
            </div>
          )}
          {tab === "claims" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              <div style={{ display: "flex", gap: 8 }}>
                <div style={{ position: "relative", flex: 1, maxWidth: 340 }}>
                  <Search size={14} color={C.textMuted} style={{ position: "absolute", left: 11, top: "50%", transform: "translateY(-50%)" }} />
                  <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search claims…" style={{ ...inputStyle, paddingLeft: 34 }} />
                </div>
                <Btn variant="secondary" size="sm" onClick={load}><RefreshCw size={13} /></Btn>
              </div>
              <ClaimsTable claims={filtered} loading={loading} onRowClick={id => setDetailId(id)} emptyTitle="No claims found" />
            </div>
          )}
        </div>
      </div>

      <Modal open={showSubmit} onClose={() => setShowSubmit(false)} title="Submit New Claim" width={620}>
        <ClaimSubmitForm onSuccess={() => { setShowSubmit(false); load(); }} onCancel={() => setShowSubmit(false)} />
      </Modal>

      {detailId && <ClaimDetailModal claimId={detailId} onClose={() => setDetailId(null)} isAdmin={false} />}
      <style>{`@media(max-width:900px){.user-stats{grid-template-columns:repeat(2,1fr)!important}}@media(max-width:480px){.user-stats{grid-template-columns:1fr!important}}`}</style>
    </div>
  );
}

/* ─── Admin App ──────────────────────────────────────────────── */
function AdminApp({ user, onLogout }) {
  const push = useToast();
  const [tab, setTab] = useState("review");
  const [pending, setPending] = useState([]);
  const [loadingPending, setLoadingPending] = useState(true);
  const [analytics, setAnalytics] = useState(null);
  const [loadingAnalytics, setLoadingAnalytics] = useState(true);
  const [detailId, setDetailId] = useState(null);
  const [search, setSearch] = useState("");
  const [mobileOpen, setMobileOpen] = useState(false);
  const [confirmApproveAll, setConfirmApproveAll] = useState(false);
  const [approvingAll, setApprovingAll] = useState(false);

  const loadPending = useCallback(async () => {
    setLoadingPending(true);
    try {
      const claims = await api.getPendingClaims();
      const withDetection = await Promise.all(claims.map(async c => {
        try { return { ...c, ...await api.runFraudDetection(c.id) }; }
        catch { return c; }
      }));
      setPending(withDetection);
    } catch { push("Failed to load claims.", "error"); }
    finally { setLoadingPending(false); }
  }, []);

  const loadAnalytics = useCallback(async () => {
    setLoadingAnalytics(true);
    try { setAnalytics(await api.getAnalytics()); }
    catch { }
    finally { setLoadingAnalytics(false); }
  }, []);

  useEffect(() => { loadPending(); loadAnalytics(); }, [loadPending, loadAnalytics]);

  const onStatusChanged = id => setPending(p => p.filter(c => c.id !== id));

  const searchFilter = c => !search || c.ref.toLowerCase().includes(search.toLowerCase()) || (c.policyNumber || "").toLowerCase().includes(search.toLowerCase()) || (c.claimantName || "").toLowerCase().includes(search.toLowerCase());

  const redClaims = pending.filter(c => c.isFraud === true && searchFilter(c));
  const greenClaims = pending.filter(c => c.isFraud === false && searchFilter(c));
  const pendingClaims = pending.filter(c => c.isFraud === null && searchFilter(c));

  const approveAllGreen = async () => {
    setApprovingAll(true); setConfirmApproveAll(false);
    try {
      await Promise.all(greenClaims.map(c => api.updateClaimStatus(c.id, "Approved")));
      setPending(p => p.filter(c => c.isFraud !== false));
      push(`${greenClaims.length} claims approved.`);
    } catch { push("Failed to approve all.", "error"); }
    finally { setApprovingAll(false); }
  };

  const navItems = [
    { key: "overview", label: "Overview", icon: LayoutDashboard },
    { key: "review", label: "Fraud Review", icon: Shield },
    { key: "statistics", label: "Statistics", icon: BarChart2 },
  ];

  return (
    <div style={{ display: "flex", height: "100vh", overflow: "hidden", background: C.bg }}>
      <Sidebar items={navItems} active={tab} onSelect={setTab} userName={user.name} role="admin" onLogout={onLogout} mobileOpen={mobileOpen} onClose={() => setMobileOpen(false)} />
      <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <TopBar
          title={tab === "overview" ? "Overview" : tab === "review" ? "Fraud Review" : "Statistics"}
          subtitle={tab === "review" && !loadingPending ? `${pending.length} claims awaiting review` : undefined}
          onMenu={() => setMobileOpen(true)}
          actions={tab === "review" ? (
            <div style={{ display: "flex", gap: 8 }}>
              {greenClaims.length > 0 && (
                <Btn variant="success" onClick={() => setConfirmApproveAll(true)} loading={approvingAll}>
                  <CheckCircle2 size={14} /> Approve {greenClaims.length} genuine
                </Btn>
              )}
              <Btn variant="secondary" onClick={loadPending}><RefreshCw size={14} /></Btn>
            </div>
          ) : undefined}
        />

        <div style={{ flex: 1, padding: 28, overflowY: "auto" }}>

          {/* Overview */}
          {tab === "overview" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 14 }} className="admin-stats">
                <StatCard label="Total Claims" value={analytics?.totalClaims ?? "—"} icon={FileText} accent={C.navy} />
                <StatCard label="Pending Review" value={analytics?.pendingCount ?? "—"} icon={Clock} color={C.amber} />
                <StatCard label="Fraud Detected" value={analytics?.fraudCount ?? "—"} icon={AlertTriangle} color={C.red} />
                <StatCard label="Approved" value={analytics?.approvedCount ?? "—"} icon={CheckCircle2} color={C.green} />
              </div>
              {analytics && (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }} className="chart-grid">
                  <ChartBox title="Fraud vs Genuine">
                    <ResponsiveContainer width="100%" height={220}>
                      <PieChart>
                        <Pie data={analytics.predictionData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => value > 0 ? `${name}: ${value}` : ""} labelLine={false} fontSize={12}>
                          {analytics.predictionData.map((d, i) => <Cell key={i} fill={d.color} />)}
                        </Pie>
                        <Tooltip contentStyle={{ fontFamily: "Inter, sans-serif", fontSize: 12.5, borderRadius: 8, border: `1px solid ${C.border}`, boxShadow: "0 4px 16px rgba(0,0,0,0.1)" }} />
                      </PieChart>
                    </ResponsiveContainer>
                  </ChartBox>
                  <ChartBox title="Claims by Risk Level">
                    <ResponsiveContainer width="100%" height={220}>
                      <BarChart data={analytics.riskData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.borderLight} vertical={false} />
                        <XAxis dataKey="name" tick={{ fontFamily: "Inter, sans-serif", fontSize: 12, fill: C.textSec }} axisLine={false} tickLine={false} />
                        <YAxis tick={{ fontFamily: "Inter, sans-serif", fontSize: 12, fill: C.textSec }} axisLine={false} tickLine={false} allowDecimals={false} />
                        <Tooltip contentStyle={{ fontFamily: "Inter, sans-serif", fontSize: 12.5, borderRadius: 8, border: `1px solid ${C.border}`, boxShadow: "0 4px 16px rgba(0,0,0,0.1)" }} />
                        <Bar dataKey="value" radius={[5, 5, 0, 0]}>
                          {analytics.riskData.map((d, i) => <Cell key={i} fill={d.color} />)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </ChartBox>
                </div>
              )}
            </div>
          )}

          {/* Fraud Review */}
          {tab === "review" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
              {/* Stats */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 14 }} className="review-stats">
                <StatCard label="Total Pending" value={pending.length} icon={Clock} accent={C.navy} />
                <StatCard label="Fraud Flagged" value={pending.filter(c => c.isFraud === true).length} icon={ShieldAlert} color={C.red} />
                <StatCard label="Likely Genuine" value={pending.filter(c => c.isFraud === false).length} icon={ShieldCheck} color={C.green} />
              </div>

              {/* Search */}
              <div style={{ position: "relative", maxWidth: 380 }}>
                <Search size={14} color={C.textMuted} style={{ position: "absolute", left: 11, top: "50%", transform: "translateY(-50%)" }} />
                <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search by claim ref, customer or policy…" style={{ ...inputStyle, paddingLeft: 34 }} />
              </div>

              {loadingPending ? (
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 12, padding: 56 }}>
                  <Spinner size={28} color={C.navy} />
                  <p style={{ fontFamily: "Inter, sans-serif", fontSize: 14, color: C.textSec, margin: 0 }}>Scanning claims for fraud signals…</p>
                </div>
              ) : pending.length === 0 ? (
                <div style={{ background: C.surface, borderRadius: 10, border: `1px solid ${C.border}` }}>
                  <EmptyState icon={CheckCircle2} title="No pending claims" subtitle="All submitted claims will appear here for fraud review." />
                </div>
              ) : (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }} className="flag-grid">
                  <FlagColumn tone="fraud" title="Fraud Flagged" subtitle="High fraud probability" claims={redClaims} onReview={id => setDetailId(id)} />
                  <FlagColumn tone="genuine" title="Likely Genuine" subtitle="Low fraud probability" claims={greenClaims} onReview={id => setDetailId(id)} onApproveAll={() => setConfirmApproveAll(true)} approvingAll={approvingAll} />
                </div>
              )}

            </div>
          )}

          {/* Statistics */}
          {tab === "statistics" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
              {loadingAnalytics ? (
                <div style={{ display: "flex", justifyContent: "center", padding: 60 }}><Spinner size={28} color={C.navy} /></div>
              ) : analytics ? (
                <>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 14 }} className="admin-stats">
                    <StatCard label="Total Claims" value={analytics.totalClaims} icon={FileText} accent={C.navy} />
                    <StatCard label="Fraud Cases" value={analytics.fraudCount} icon={AlertTriangle} color={C.red} />
                    <StatCard label="Approved" value={analytics.approvedCount} icon={CheckCircle2} color={C.green} />
                    <StatCard label="Rejected" value={analytics.rejectedCount} icon={XCircle} color={C.textSec} />
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }} className="chart-grid">
                    <ChartBox title="Fraud vs Genuine Breakdown">
                      <ResponsiveContainer width="100%" height={250}>
                        <PieChart>
                          <Pie data={analytics.predictionData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={95} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`} fontSize={12}>
                            {analytics.predictionData.map((d, i) => <Cell key={i} fill={d.color} />)}
                          </Pie>
                          <Tooltip contentStyle={{ fontFamily: "Inter, sans-serif", fontSize: 12.5, borderRadius: 8, border: `1px solid ${C.border}` }} />
                        </PieChart>
                      </ResponsiveContainer>
                    </ChartBox>
                    <ChartBox title="Risk Distribution">
                      <ResponsiveContainer width="100%" height={250}>
                        <BarChart data={analytics.riskData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke={C.borderLight} vertical={false} />
                          <XAxis dataKey="name" tick={{ fontFamily: "Inter, sans-serif", fontSize: 13, fill: C.textSec }} axisLine={false} tickLine={false} />
                          <YAxis tick={{ fontFamily: "Inter, sans-serif", fontSize: 13, fill: C.textSec }} axisLine={false} tickLine={false} allowDecimals={false} />
                          <Tooltip contentStyle={{ fontFamily: "Inter, sans-serif", fontSize: 12.5, borderRadius: 8, border: `1px solid ${C.border}` }} />
                          <Bar dataKey="value" radius={[5, 5, 0, 0]}>
                            {analytics.riskData.map((d, i) => <Cell key={i} fill={d.color} />)}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </ChartBox>
                  </div>
                </>
              ) : (
                <EmptyState icon={BarChart2} title="No data available" subtitle="Statistics will appear once claims are submitted and analyzed." />
              )}
            </div>
          )}
        </div>
      </div>

      {detailId && <ClaimDetailModal claimId={detailId} onClose={() => setDetailId(null)} onStatusChanged={onStatusChanged} isAdmin />}
      <ConfirmDialog open={confirmApproveAll} onClose={() => setConfirmApproveAll(false)} onConfirm={approveAllGreen} title="Approve all genuine claims?"
        message={`This will approve ${greenClaims.length} claim${greenClaims.length !== 1 ? "s" : ""} flagged as likely genuine. This action cannot be undone.`}
        confirmLabel={`Approve ${greenClaims.length} claims`} loading={approvingAll} />

      <style>{`
        @media(max-width:900px){.admin-stats{grid-template-columns:repeat(2,1fr)!important}.chart-grid{grid-template-columns:1fr!important}.review-stats{grid-template-columns:repeat(2,1fr)!important}.flag-grid{grid-template-columns:1fr!important}}
        @media(max-width:480px){.admin-stats{grid-template-columns:1fr!important}.review-stats{grid-template-columns:1fr!important}}
      `}</style>
    </div>
  );
}

function ChartBox({ title, children }) {
  return (
    <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, padding: "20px 24px", boxShadow: "0 1px 4px rgba(13,27,42,0.06)" }}>
      <h3 style={{ fontFamily: "Inter, sans-serif", fontWeight: 700, fontSize: 14, color: C.text, margin: "0 0 18px" }}>{title}</h3>
      {children}
    </div>
  );
}

/* ─── Login ──────────────────────────────────────────────────── */
function LoginPage({ onLogin }) {
  const push = useToast();
  const [role, setRole] = useState("user");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const submit = async e => {
    e.preventDefault();
    setError(""); setLoading(true);
    try { onLogin(await api.loginUser(email, password, role)); }
    catch (err) { setError(err.message || "Sign in failed."); }
    finally { setLoading(false); }
  };

  return (
    <div style={{ minHeight: "100vh", background: "linear-gradient(135deg, #0D1B2A 0%, #1B3A5C 60%, #0E3A5C 100%)", display: "flex", alignItems: "center", justifyContent: "center", padding: 20 }}>
      <div style={{ width: "100%", maxWidth: 420 }}>
        {/* Brand */}
        <div style={{ textAlign: "center", marginBottom: 32 }}>
          <div style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", width: 52, height: 52, borderRadius: 14, background: C.teal, marginBottom: 14, boxShadow: "0 8px 24px rgba(14,116,144,0.35)" }}>
            <Shield size={26} color="#FFF" />
          </div>
          <h1 style={{ fontFamily: "Inter, sans-serif", fontWeight: 800, fontSize: 22, color: "#FFF", margin: "0 0 6px", letterSpacing: "0.01em" }}>CLAIM VISION</h1>
          <p style={{ fontFamily: "Inter, sans-serif", fontSize: 13.5, color: "rgba(255,255,255,0.5)", margin: 0 }}>Vehicle Insurance Management System</p>
        </div>

        {/* Card */}
        <div style={{ background: "#FFF", borderRadius: 14, padding: 32, boxShadow: "0 24px 64px rgba(0,0,0,0.25)" }}>
          <h2 style={{ fontFamily: "Inter, sans-serif", fontWeight: 700, fontSize: 18, color: C.text, margin: "0 0 20px" }}>Sign in to your account</h2>

          {/* Role tabs */}
          <div style={{ display: "flex", background: C.bg, borderRadius: 8, padding: 3, marginBottom: 24, border: `1px solid ${C.border}` }}>
            {[["user", "User"], ["admin", "Administrator"]].map(([k, label]) => (
              <button key={k} onClick={() => { setRole(k); setError(""); }}
                style={{ flex: 1, padding: "8px 0", borderRadius: 6, border: "none", cursor: "pointer", fontFamily: "Inter, sans-serif", fontSize: 13.5, fontWeight: 700, background: role === k ? C.navy : "transparent", color: role === k ? "#FFF" : C.textSec, boxShadow: role === k ? "0 2px 8px rgba(13,27,42,0.2)" : "none", transition: "all 0.15s" }}>
                {label}
              </button>
            ))}
          </div>

          <form onSubmit={submit} style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <Field label="Email address" required>
              <input type="text" value={email} onChange={e => setEmail(e.target.value)} placeholder="Enter your email or username" required
                style={{ ...inputStyle, height: 42, fontSize: 14 }} />
            </Field>
            <Field label="Password" required>
              <div style={{ position: "relative" }}>
                <input type={showPw ? "text" : "password"} value={password} onChange={e => setPassword(e.target.value)} placeholder="Enter your password" required
                  style={{ ...inputStyle, height: 42, fontSize: 14, paddingRight: 42 }} />
                <button type="button" onClick={() => setShowPw(v => !v)}
                  style={{ position: "absolute", right: 12, top: "50%", transform: "translateY(-50%)", background: "none", border: "none", cursor: "pointer", color: C.textMuted, display: "flex" }}>
                  {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </Field>

            {error && (
              <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "10px 14px", background: C.redBg, border: `1px solid ${C.redBorder}`, borderRadius: 7 }}>
                <AlertTriangle size={14} color={C.red} />
                <span style={{ fontFamily: "Inter, sans-serif", fontSize: 13, color: C.red }}>{error}</span>
              </div>
            )}

            <button type="submit" disabled={loading}
              style={{ width: "100%", padding: "12px 0", marginTop: 4, background: loading ? C.navyMed : C.navy, color: "#FFF", border: "none", borderRadius: 8, fontSize: 15, fontWeight: 700, fontFamily: "Inter, sans-serif", cursor: loading ? "not-allowed" : "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 8, transition: "background 0.15s", boxShadow: "0 2px 8px rgba(13,27,42,0.2)" }}>
              {loading && <Spinner size={16} />}
              {loading ? "Signing in…" : "Sign in"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

/* ─── Root ───────────────────────────────────────────────────── */
export default function ClaimSeal() {
  const [user, setUser] = useState(null);
  return (
    <ToastProvider>
      <style>{`
        *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
        body{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:${C.bg};color:${C.text};-webkit-font-smoothing:antialiased}
        input,textarea,button,select{font-family:inherit}
        input:focus,textarea:focus{outline:none;border-color:${C.navy}!important;box-shadow:0 0 0 3px rgba(13,27,42,0.08)}
        ::-webkit-scrollbar{width:5px;height:5px}
        ::-webkit-scrollbar-track{background:transparent}
        ::-webkit-scrollbar-thumb{background:${C.border};border-radius:3px}
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;600&display=swap');
      `}</style>
      {!user
        ? <LoginPage onLogin={d => setUser(d.user)} />
        : user.role === "admin"
          ? <AdminApp user={user} onLogout={() => { clearToken(); _userName = ""; setUser(null); }} />
          : <UserApp user={user} onLogout={() => { clearToken(); _userName = ""; setUser(null); }} />
      }
    </ToastProvider>
  );
}
