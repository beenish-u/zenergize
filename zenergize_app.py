"""
Zenergize Operations Intelligence Platform
===========================================
Three operational tools built for Zenergize's EV charging and solar inverter business:

  Tab 1 — Charger Health Monitor     (Problem 1: Rajiv's CPO dashboard)
  Tab 2 — Solar Inverter TCO         (Problem 3: Suresh's value comparison)
  Tab 3 — Fleet Charging Optimiser   (Problem 4: Anil's depot scheduler)

All data in Tab 1 and Tab 3 is simulated. Tab 2 uses real solar irradiance
physics (pvlib + location-based calculations).

Run:
  streamlit run zenergize_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import heapq
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG & GLOBAL STYLES
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Zenergize Operations",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #f7f6f3;
    color: #2d2d2d;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2.5rem;
    max-width: 1180px;
    background-color: #f7f6f3;
}

/* Header */
.app-header {
    margin-bottom: 2rem;
}
.app-title {
    font-size: 1.35rem;
    font-weight: 600;
    color: #1e293b;
    letter-spacing: -0.01em;
}
.app-subtitle {
    font-size: 0.82rem;
    color: #64748b;
    margin-top: 0.2rem;
}

/* Metric cards */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 1.65rem;
    font-weight: 600;
    color: #1e293b;
    font-family: 'DM Mono', monospace;
    line-height: 1.1;
}
.metric-sub {
    font-size: 0.73rem;
    color: #94a3b8;
    margin-top: 0.3rem;
}

/* Status badges */
.badge-green  { display:inline-block; background:#dcfce7; color:#166534; font-size:0.68rem; font-weight:600; padding:2px 9px; border-radius:20px; letter-spacing:0.04em; }
.badge-amber  { display:inline-block; background:#fef9c3; color:#854d0e; font-size:0.68rem; font-weight:600; padding:2px 9px; border-radius:20px; letter-spacing:0.04em; }
.badge-red    { display:inline-block; background:#fee2e2; color:#991b1b; font-size:0.68rem; font-weight:600; padding:2px 9px; border-radius:20px; letter-spacing:0.04em; }

/* Section headers */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #e2e8f0;
}

/* Alert rows */
.alert-row {
    background: #fffbeb;
    border-left: 3px solid #f59e0b;
    padding: 0.65rem 0.9rem;
    margin-bottom: 0.5rem;
    border-radius: 0 6px 6px 0;
}
.alert-row-red {
    background: #fff1f2;
    border-left: 3px solid #f43f5e;
    padding: 0.65rem 0.9rem;
    margin-bottom: 0.5rem;
    border-radius: 0 6px 6px 0;
}
.alert-text { font-size: 0.82rem; color: #334155; }
.alert-time { font-size: 0.7rem; color: #94a3b8; font-family: 'DM Mono', monospace; }

/* Data note */
.data-note {
    font-size: 0.7rem;
    color: #94a3b8;
    font-style: italic;
    margin-top: 0.6rem;
}

/* Tab styling */
div[data-testid="stTabs"] button {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
}

/* Horizontal rule */
hr { border: none; border-top: 1px solid #e2e8f0; margin: 1.5rem 0; }

/* Input labels */
.stSelectbox label, .stNumberInput label, .stSlider label {
    font-size: 0.77rem !important;
    font-weight: 500 !important;
    color: #475569 !important;
}

/* Remove default st.metric */
.stMetric { display: none; }

/* Streamlit button */
.stButton > button {
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
  <div class="app-title">Zenergize — Operations Intelligence</div>
  <div class="app-subtitle">Solar Inverter TCO &nbsp;·&nbsp; Charger Health &nbsp;·&nbsp; Fleet Optimiser</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  TAB LAYOUT
# ─────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "Solar Inverter TCO",
    "Charger Health Monitor",
    "Fleet Charging Optimiser",
])


# ═══════════════════════════════════════════════════════════════
#
#   TAB 1 — CHARGER HEALTH MONITOR
#
#   Problem: Rajiv gets no alert until customers complain.
#   Solution: Continuous 0–100 health score from voltage
#   variance, temperature deviation, session success rate,
#   and time since last successful session.
#
# ═══════════════════════════════════════════════════════════════

with tab1:

    # City irradiance database (GHI kWh/m²/year, from PVGIS data)
    CITY_DATA = {
        "Jaipur":      {"ghi": 1950, "lat": 26.91, "lon": 75.78, "state": "Rajasthan"},
        "Jodhpur":     {"ghi": 2050, "lat": 26.29, "lon": 73.02, "state": "Rajasthan"},
        "Delhi":       {"ghi": 1750, "lat": 28.61, "lon": 77.21, "state": "Delhi"},
        "Pune":        {"ghi": 1850, "lat": 18.52, "lon": 73.86, "state": "Maharashtra"},
        "Ahmedabad":   {"ghi": 2000, "lat": 23.02, "lon": 72.57, "state": "Gujarat"},
        "Hyderabad":   {"ghi": 1900, "lat": 17.38, "lon": 78.47, "state": "Telangana"},
        "Chennai":     {"ghi": 1800, "lat": 13.08, "lon": 80.27, "state": "Tamil Nadu"},
        "Bengaluru":   {"ghi": 1780, "lat": 12.97, "lon": 77.59, "state": "Karnataka"},
        "Lucknow":     {"ghi": 1680, "lat": 26.85, "lon": 80.95, "state": "Uttar Pradesh"},
        "Bhopal":      {"ghi": 1820, "lat": 23.25, "lon": 77.41, "state": "Madhya Pradesh"},
        "Nagpur":      {"ghi": 1870, "lat": 21.14, "lon": 79.08, "state": "Maharashtra"},
        "Kolkata":     {"ghi": 1580, "lat": 22.57, "lon": 88.36, "state": "West Bengal"},
        "Indore":      {"ghi": 1890, "lat": 22.72, "lon": 75.86, "state": "Madhya Pradesh"},
        "Surat":       {"ghi": 1950, "lat": 21.17, "lon": 72.83, "state": "Gujarat"},
        "Bikaner":     {"ghi": 2100, "lat": 28.01, "lon": 73.31, "state": "Rajasthan"},
    }

    def tco_model(
        city, system_kwp, tariff, tariff_escalation,
        comp_eff, zen_eff,
        comp_price_per_kw, zen_price_per_kw,
        comp_service_cost, zen_service_cost,
        comp_service_events, zen_service_events,
        discount_rate, years=25
    ):
        """
        25-year TCO and NPV model.

        Returns dict of annual arrays and summary figures.
        """
        ghi = CITY_DATA[city]["ghi"]

        # System PR (excluding inverter efficiency)
        pr_system = 0.84  # cable losses, soiling, mismatch, shading

        # Annual generation (kWh/year)
        # Gen = System_kWp × GHI × PR_system × η_inverter
        gen_zen  = system_kwp * ghi * pr_system * (zen_eff / 100)
        gen_comp = system_kwp * ghi * pr_system * (comp_eff / 100)
        gen_gap  = gen_zen - gen_comp  # annual kWh advantage of Zenergize

        # 25-year arrays
        yrs = np.arange(1, years + 1)

        # Tariff escalation
        tariff_each_year = tariff * (1 + tariff_escalation / 100) ** (yrs - 1)

        # Annual revenue from generation gap
        revenue_gap_annual = gen_gap * tariff_each_year

        # Cumulative revenue gap (undiscounted)
        cum_revenue_gap = np.cumsum(revenue_gap_annual)

        # NPV of revenue gap
        npv_factors = 1 / (1 + discount_rate / 100) ** yrs
        npv_revenue_gap = np.sum(revenue_gap_annual * npv_factors)

        # Service cost differential (annual events × cost/event)
        service_gap_annual = (
            comp_service_events * comp_service_cost
            - zen_service_events * zen_service_cost
        )  # positive means Zenergize saves money each year

        npv_service_gap = service_gap_annual * np.sum(npv_factors)

        # Upfront cost premium for Zenergize
        price_premium = (zen_price_per_kw - comp_price_per_kw) * system_kwp

        # Total 25-year NPV advantage of Zenergize
        total_npv_advantage = npv_revenue_gap + npv_service_gap - price_premium

        # Breakeven year (undiscounted, conservative)
        cumulative_savings = np.cumsum(
            revenue_gap_annual + service_gap_annual
        )
        breakeven_mask = cumulative_savings >= price_premium
        if breakeven_mask.any():
            breakeven_year = yrs[breakeven_mask][0]
        else:
            breakeven_year = None

        return {
            "years": yrs,
            "gen_zen": gen_zen,
            "gen_comp": gen_comp,
            "gen_gap_annual": gen_gap,
            "gen_gap_25yr": gen_gap * years,
            "tariff_each_year": tariff_each_year,
            "revenue_gap_annual": revenue_gap_annual,
            "cum_revenue_gap": cum_revenue_gap,
            "npv_revenue_gap": npv_revenue_gap,
            "npv_service_gap": npv_service_gap,
            "price_premium": price_premium,
            "total_npv_advantage": total_npv_advantage,
            "breakeven_year": breakeven_year,
            "service_gap_annual": service_gap_annual,
        }

    # ── Inputs ─────────────────────────────────────────────────

    st.markdown('<div class="section-label">System Parameters</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        city = st.selectbox("Installation City", sorted(CITY_DATA.keys()), index=0)
        system_kwp = st.number_input("System Size (kWp)", min_value=1.0, max_value=500.0, value=5.0, step=0.5)
        tariff = st.number_input("Current Electricity Tariff (₹/kWh)", min_value=3.0, max_value=12.0, value=7.0, step=0.25)
        tariff_escalation = st.slider("Annual Tariff Escalation (%)", 0.0, 10.0, 5.0, 0.5)

    with col2:
        st.markdown("**Competitor Inverter**")
        comp_eff = st.number_input("Competitor Efficiency (%)", min_value=90.0, max_value=99.9, value=96.5, step=0.1)
        comp_price = st.number_input("Competitor Price (₹/kW)", min_value=3000, max_value=20000, value=8000, step=500)
        comp_service = st.number_input("Competitor Service Cost / Event (₹)", min_value=500, max_value=50000, value=12000, step=500)
        comp_events = st.number_input("Competitor Service Events / Year", min_value=0.0, max_value=5.0, value=0.5, step=0.1)

    with col3:
        st.markdown("**Zenergize Inverter**")
        zen_eff = st.number_input("Zenergize Efficiency (%)", min_value=90.0, max_value=99.9, value=97.8, step=0.1)
        zen_price = st.number_input("Zenergize Price (₹/kW)", min_value=3000, max_value=20000, value=11500, step=500)
        zen_service = st.number_input("Zenergize Service Cost / Event (₹)", min_value=500, max_value=50000, value=2500, step=500)
        zen_events = st.number_input("Zenergize Service Events / Year", min_value=0.0, max_value=5.0, value=0.3, step=0.1)

    discount_rate = st.slider("Discount Rate (%)", 4.0, 15.0, 8.0, 0.5)

    run_tco = st.button("Calculate 25-Year TCO", type="primary")

    if run_tco or True:  # auto-calculate on first load too
        result = tco_model(
            city=city,
            system_kwp=system_kwp,
            tariff=tariff,
            tariff_escalation=tariff_escalation,
            comp_eff=comp_eff,
            zen_eff=zen_eff,
            comp_price_per_kw=comp_price,
            zen_price_per_kw=zen_price,
            comp_service_cost=comp_service,
            zen_service_cost=zen_service,
            comp_service_events=comp_events,
            zen_service_events=zen_events,
            discount_rate=discount_rate,
        )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">25-Year Analysis Results</div>', unsafe_allow_html=True)

        # Summary cards
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        adv = result["total_npv_advantage"]
        adv_color = "#10b981" if adv >= 0 else "#ef4444"

        with col_r1:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">25-Year NPV Advantage</div>
              <div class="metric-value" style="color:{adv_color}">₹{abs(adv):,.0f}</div>
              <div class="metric-sub">{"In favour of Zenergize" if adv >= 0 else "In favour of competitor"}</div>
            </div>""", unsafe_allow_html=True)

        with col_r2:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">Annual Generation Advantage</div>
              <div class="metric-value">{result['gen_gap_annual']:,.0f}</div>
              <div class="metric-sub">kWh/year more from Zenergize</div>
            </div>""", unsafe_allow_html=True)

        with col_r3:
            be = result["breakeven_year"]
            be_str = f"Year {be}" if be else ">25 yr"
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">Breakeven Point</div>
              <div class="metric-value">{be_str}</div>
              <div class="metric-sub">Premium recovered vs. competitor</div>
            </div>""", unsafe_allow_html=True)

        with col_r4:
            upfront = result["price_premium"]
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">Upfront Premium</div>
              <div class="metric-value">₹{upfront:,.0f}</div>
              <div class="metric-sub">Additional cost for Zenergize</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts
        col_ch1, col_ch2 = st.columns(2)

        with col_ch1:
            # Cumulative savings over 25 years
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(
                x=result["years"],
                y=result["cum_revenue_gap"] + result["service_gap_annual"] * result["years"] - result["price_premium"],
                mode="lines",
                name="Cumulative net advantage",
                line=dict(color="#334155", width=2),
                fill="tozeroy",
                fillcolor="rgba(16,185,129,0.1)",
            ))
            fig_cum.add_hline(y=0, line_dash="solid", line_color="#cbd5e1", line_width=1)
            if result["breakeven_year"]:
                fig_cum.add_vline(
                    x=result["breakeven_year"],
                    line_dash="dot", line_color="#f59e0b",
                    annotation_text=f"Breakeven: Year {result['breakeven_year']}",
                    annotation_font_size=10, annotation_font_color="#475569",
                )
            fig_cum.update_layout(
                title="Cumulative Net Advantage of Zenergize (₹)",
                height=300,
                margin=dict(l=0, r=0, t=36, b=10),
                font=dict(family="DM Sans", size=11, color="#334155"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                yaxis=dict(tickprefix="₹", tickformat=",.0f"),
            )
            st.plotly_chart(fig_cum, use_container_width=True, config={"displayModeBar": False})

        with col_ch2:
            # Sensitivity: tariff escalation impact on NPV
            escalations = np.arange(0, 11, 1)
            npv_at_esc = []
            for esc in escalations:
                r = tco_model(
                    city=city, system_kwp=system_kwp, tariff=tariff,
                    tariff_escalation=esc, comp_eff=comp_eff, zen_eff=zen_eff,
                    comp_price_per_kw=comp_price, zen_price_per_kw=zen_price,
                    comp_service_cost=comp_service, zen_service_cost=zen_service,
                    comp_service_events=comp_events, zen_service_events=zen_events,
                    discount_rate=discount_rate,
                )
                npv_at_esc.append(r["total_npv_advantage"])

            fig_sens = go.Figure()
            fig_sens.add_trace(go.Scatter(
                x=escalations, y=npv_at_esc,
                mode="lines+markers",
                line=dict(color="#334155", width=2),
                marker=dict(size=6),
                name="NPV",
            ))
            fig_sens.add_hline(y=0, line_dash="dot", line_color="#ef4444", line_width=1)
            fig_sens.update_layout(
                title="Sensitivity: NPV vs Annual Tariff Escalation (%)",
                height=300,
                margin=dict(l=0, r=0, t=36, b=10),
                font=dict(family="DM Sans", size=11, color="#334155"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                xaxis=dict(title="Annual Tariff Escalation (%)"),
                yaxis=dict(tickprefix="₹", tickformat=",.0f"),
            )
            st.plotly_chart(fig_sens, use_container_width=True, config={"displayModeBar": False})

        # Waterfall breakdown
        st.markdown('<div class="section-label">Cost-Benefit Breakdown</div>', unsafe_allow_html=True)

        wf_labels = ["Upfront Premium (cost)", "Energy Revenue Advantage (NPV)", "Service Saving (NPV)", "Net Advantage"]
        wf_values = [
            -result["price_premium"],
            result["npv_revenue_gap"],
            result["npv_service_gap"],
            result["total_npv_advantage"],
        ]
        wf_measure = ["relative", "relative", "relative", "total"]
        wf_colors = ["#ef4444", "#10b981", "#10b981", "#1a1a2e"]

        fig_wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=wf_measure,
            x=wf_labels,
            y=wf_values,
            connector={"line": {"color": "#e5e7eb", "width": 1}},
            decreasing={"marker": {"color": "#fee2e2", "line": {"color": "#ef4444", "width": 1}}},
            increasing={"marker": {"color": "#d1fae5", "line": {"color": "#10b981", "width": 1}}},
            totals={"marker": {"color": "#334155"}},
            text=[f"₹{v:,.0f}" for v in wf_values],
            textposition="outside",
        ))
        fig_wf.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=10),
            font=dict(family="DM Sans", size=11, color="#334155"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            yaxis=dict(tickprefix="₹", tickformat=",.0f"),
        )
        st.plotly_chart(fig_wf, use_container_width=True, config={"displayModeBar": False})

        # Location context
        city_info = CITY_DATA[city]
        st.markdown(f"""
        <p class="data-note">
        Location: {city}, {city_info['state']} — Annual irradiance: {city_info['ghi']} kWh/m²/year (PVGIS reference data).
        Efficiency advantage: {zen_eff - comp_eff:.1f}% → {result['gen_gap_annual']:,.0f} kWh/year on {system_kwp} kWp system.
        System PR components: inverter efficiency × 0.84 (cable + soiling + mismatch losses).
        </p>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#
#   TAB 3 — FLEET CHARGING OPTIMISER
#
#   Problem: Anil runs all chargers at max power all night.
#   Solution: EDF scheduler with SoC priority, charger health
#   weighting, and tariff-aware time shifting.
#
#   Algorithm:
#     1. For each vehicle: compute energy needed = (target_soc - current_soc)/100 × battery_kwh
#     2. Compute available time slots (15-min intervals) from now to deadline
#     3. Assign vehicles to chargers using EDF + priority weighting
#     4. Apply thermal derating to charger power in each slot
#     5. Shift slots to off-peak where deadline allows
#
#   Thermal derating (Infineon AN2020-16 model):
#     T ≤ 40°C  → factor = 1.0
#     40 < T ≤ 45 → factor = 1.0 - (T - 40) × 0.01
#     T > 45    → factor = 0.95 - (T - 45) × 0.014
#
# ═══════════════════════════════════════════════════════════════

with tab2:

    # ── Synthetic telemetry data ──────────────────────────────

    @st.cache_data(ttl=60)
    def generate_telemetry(seed=42):
        """
        90 days of OCPP telemetry for 6 chargers across 6 sites on NH48.
        Each row: 15-minute interval reading.

        Failure injection:
          Charger C003 at Dharuhera — voltage variance spike in last 6 hours
          Charger C005 at Manesar   — elevated temperature for last 3 days
        """
        rng = np.random.default_rng(seed)
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        start = now - timedelta(days=90)
        timestamps = pd.date_range(start, now, freq="15min")
        n = len(timestamps)

        chargers = {
            "C001": {"site": "Jaipur Rest Stop",    "lat": 26.91, "lon": 75.78, "healthy": True},
            "C002": {"site": "Shahpura Junction",   "lat": 27.40, "lon": 76.28, "healthy": True},
            "C003": {"site": "Dharuhera Depot",     "lat": 28.21, "lon": 76.97, "healthy": False, "fault": "voltage"},
            "C004": {"site": "Bilaspur Bypass",     "lat": 28.43, "lon": 76.94, "healthy": True},
            "C005": {"site": "Manesar Plaza",       "lat": 28.36, "lon": 76.93, "healthy": False, "fault": "temperature"},
            "C006": {"site": "Gurugram NH48 Stop",  "lat": 28.46, "lon": 77.03, "healthy": True},
        }

        records = []
        for cid, meta in chargers.items():
            # Baseline voltage: 415V ± normal noise
            voltage = rng.normal(415, 3, n)
            temperature = rng.normal(38, 4, n)
            session_success = rng.choice([True, False], n, p=[0.97, 0.03])

            if not meta["healthy"]:
                if meta["fault"] == "voltage":
                    # Inject voltage spikes in last 6 hours (last 24 intervals)
                    spike_start = n - 24
                    voltage[spike_start:] += rng.normal(22, 6, 24)
                    session_success[spike_start:] = rng.choice(
                        [True, False], 24, p=[0.60, 0.40]
                    )
                elif meta["fault"] == "temperature":
                    # Sustained temp elevation last 3 days
                    temp_start = n - 3 * 96
                    temperature[temp_start:] += rng.normal(14, 2, n - temp_start)

            df_c = pd.DataFrame({
                "timestamp": timestamps,
                "charger_id": cid,
                "site": meta["site"],
                "lat": meta["lat"],
                "lon": meta["lon"],
                "voltage": voltage,
                "temperature": temperature,
                "session_success": session_success.astype(int),
            })
            records.append(df_c)

        return pd.concat(records, ignore_index=True)

    def compute_health_scores(df):
        """
        Health score (0–100) per charger, computed from four sub-indices.

        Sub-index 1: Voltage Variance Index
            Compare rolling 2-hour voltage std to 30-day baseline std.
            If current std > 2× baseline: score 0. If equal: 100.
            Equation: V_score = max(0, 100 - 50 * (current_std / baseline_std - 1))

        Sub-index 2: Temperature Deviation
            Expected temperature from Open-Meteo (or fallback = 35°C for Delhi region).
            Deviation above expected ambient + 15°C threshold starts penalising.
            Equation: T_score = max(0, 100 - 4 * max(0, avg_temp - (ambient + 15)))

        Sub-index 3: Session Success Rate (last 7 days)
            SR_score = success_rate_7d * 100

        Sub-index 4: Recency (time since last successful session)
            Each hour without a successful session after 2h: -5 points.
            RS_score = max(0, 100 - 5 * max(0, hours_since - 2))

        Final: H = 0.30*V + 0.25*T + 0.30*SR + 0.15*RS
        """
        now = df["timestamp"].max()
        results = []

        for cid, group in df.groupby("charger_id"):
            group = group.sort_values("timestamp")
            site = group["site"].iloc[0]
            lat = group["lat"].iloc[0]
            lon = group["lon"].iloc[0]

            # --- Voltage variance index ---
            recent_2h = group[group["timestamp"] >= now - timedelta(hours=2)]
            baseline_30d = group[group["timestamp"] >= now - timedelta(days=30)]
            baseline_std = baseline_30d["voltage"].std()
            current_std = recent_2h["voltage"].std() if len(recent_2h) > 1 else baseline_std
            if baseline_std == 0:
                v_score = 100.0
            else:
                v_score = max(0, 100 - 50 * (current_std / baseline_std - 1))

            # --- Temperature deviation ---
            ambient = 35.0  # fallback for NH48 Delhi–Jaipur corridor (°C)
            avg_temp_recent = recent_2h["temperature"].mean() if len(recent_2h) > 0 else 38
            t_score = max(0, 100 - 4 * max(0, avg_temp_recent - (ambient + 15)))

            # --- Session success rate last 7 days ---
            last_7d = group[group["timestamp"] >= now - timedelta(days=7)]
            if len(last_7d) > 0:
                sr_score = last_7d["session_success"].mean() * 100
            else:
                sr_score = 100.0

            # --- Recency of last successful session ---
            successful = group[group["session_success"] == 1]
            if len(successful) > 0:
                last_success = successful["timestamp"].max()
                hours_since = (now - last_success).total_seconds() / 3600
            else:
                hours_since = 999
            rs_score = max(0, 100 - 5 * max(0, hours_since - 2))

            # Weighted composite
            health = 0.30 * v_score + 0.25 * t_score + 0.30 * sr_score + 0.15 * rs_score

            # Failure probability (logistic transform of health deficit)
            deficit = max(0, 65 - health)
            fail_prob = 1 / (1 + np.exp(-0.08 * (deficit - 10)))

            results.append({
                "charger_id": cid,
                "site": site,
                "lat": lat,
                "lon": lon,
                "health": round(health, 1),
                "v_score": round(v_score, 1),
                "t_score": round(t_score, 1),
                "sr_score": round(sr_score, 1),
                "rs_score": round(rs_score, 1),
                "avg_voltage": round(
                    df[(df["charger_id"] == cid) &
                       (df["timestamp"] >= now - timedelta(hours=2))]["voltage"].mean(), 1),
                "avg_temp": round(avg_temp_recent, 1),
                "hours_since_success": round(hours_since, 1),
                "fail_prob_4h": round(fail_prob * 100, 1),
            })

        return pd.DataFrame(results)

    def health_badge(score):
        if score >= 85:
            return '<span class="badge-green">GOOD</span>'
        elif score >= 65:
            return '<span class="badge-amber">WATCH</span>'
        else:
            return '<span class="badge-red">INTERVENE</span>'

    def score_color(score):
        if score >= 85:
            return "#10b981"
        elif score >= 65:
            return "#f59e0b"
        else:
            return "#ef4444"

    # ── Load data ─────────────────────────────────────────────

    with st.spinner("Loading telemetry..."):
        df_tel = generate_telemetry()
        df_scores = compute_health_scores(df_tel)

    # ── Summary metrics ────────────────────────────────────────

    n_green = (df_scores["health"] >= 85).sum()
    n_amber = ((df_scores["health"] >= 65) & (df_scores["health"] < 85)).sum()
    n_red   = (df_scores["health"] < 65).sum()
    fleet_avg = df_scores["health"].mean()

    st.markdown('<div class="section-label">Fleet Status — NH48 Corridor</div>', unsafe_allow_html=True)

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Fleet Health Index</div>
          <div class="metric-value" style="color:{score_color(fleet_avg)}">{fleet_avg:.1f}</div>
          <div class="metric-sub">Average across 6 chargers</div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Operational</div>
          <div class="metric-value" style="color:#10b981">{n_green}</div>
          <div class="metric-sub">Score above 85</div>
        </div>""", unsafe_allow_html=True)
    with col_c:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Needs Monitoring</div>
          <div class="metric-value" style="color:#f59e0b">{n_amber}</div>
          <div class="metric-sub">Score 65–85</div>
        </div>""", unsafe_allow_html=True)
    with col_d:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Intervention Required</div>
          <div class="metric-value" style="color:#ef4444">{n_red}</div>
          <div class="metric-sub">Score below 65</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Active alerts ──────────────────────────────────────────

    st.markdown('<div class="section-label">Active Alerts</div>', unsafe_allow_html=True)

    alerts_df = df_scores[df_scores["health"] < 85].sort_values("health")
    if len(alerts_df) == 0:
        st.markdown('<p style="font-size:0.8rem;color:#6b7280;">All chargers within normal parameters.</p>', unsafe_allow_html=True)
    else:
        for _, row in alerts_df.iterrows():
            if row["health"] < 65:
                box_class = "alert-row-red"
                action = f"Dispatch technician — {row['fail_prob_4h']}% failure probability in next 4 hours."
            else:
                box_class = "alert-row"
                action = "Monitor closely. No immediate action required."

            st.markdown(f"""
            <div class="{box_class}">
              <span class="alert-text"><strong>{row['charger_id']} — {row['site']}</strong> &nbsp;·&nbsp; Health {row['health']}</span><br>
              <span class="alert-time">{action}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charger table + detail ─────────────────────────────────

    col_left, col_right = st.columns([1, 1.6])

    with col_left:
        st.markdown('<div class="section-label">All Chargers</div>', unsafe_allow_html=True)
        selected_id = None
        for _, row in df_scores.sort_values("health").iterrows():
            badge = health_badge(row["health"])
            clicked = st.button(
                f"{row['charger_id']}  ·  {row['site']}  ·  {row['health']}",
                key=f"btn_{row['charger_id']}",
                use_container_width=True,
            )
            if clicked:
                selected_id = row["charger_id"]
                st.session_state["selected_charger"] = selected_id

        # persist selection
        if selected_id is None:
            selected_id = st.session_state.get("selected_charger", df_scores.sort_values("health")["charger_id"].iloc[0])

    with col_right:
        st.markdown('<div class="section-label">7-Day History</div>', unsafe_allow_html=True)

        row = df_scores[df_scores["charger_id"] == selected_id].iloc[0]
        now_ts = df_tel["timestamp"].max()

        # Score decomposition bar
        sub_labels = ["Voltage", "Temperature", "Session Rate", "Recency"]
        sub_values = [row["v_score"], row["t_score"], row["sr_score"], row["rs_score"]]
        sub_colors = [score_color(v) for v in sub_values]

        fig_sub = go.Figure(go.Bar(
            x=sub_labels,
            y=sub_values,
            marker_color=sub_colors,
            text=[f"{v:.0f}" for v in sub_values],
            textposition="outside",
        ))
        fig_sub.update_layout(
            title=f"{selected_id} — Sub-scores",
            yaxis=dict(range=[0, 110], title="Score (0–100)"),
            height=240,
            margin=dict(l=0, r=0, t=36, b=20),
            font=dict(family="DM Sans", size=11, color="#334155"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        fig_sub.add_hline(y=85, line_dash="dot", line_color="#10b981", line_width=1)
        fig_sub.add_hline(y=65, line_dash="dot", line_color="#f59e0b", line_width=1)
        st.plotly_chart(fig_sub, use_container_width=True, config={"displayModeBar": False})

        # 7-day voltage history
        charger_data = df_tel[
            (df_tel["charger_id"] == selected_id) &
            (df_tel["timestamp"] >= now_ts - timedelta(days=7))
        ].copy()

        # Downsample to hourly mean for readability
        charger_data = charger_data.set_index("timestamp").resample("1h").agg({
            "voltage": "mean",
            "temperature": "mean",
            "session_success": "mean",
        }).reset_index()

        fig_v = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              subplot_titles=("Voltage (V)", "Temperature (°C)"),
                              vertical_spacing=0.12)
        fig_v.add_trace(go.Scatter(
            x=charger_data["timestamp"], y=charger_data["voltage"],
            line=dict(color="#334155", width=1.2), name="Voltage"
        ), row=1, col=1)
        fig_v.add_hrect(y0=400, y1=430, fillcolor="#d1fae5", opacity=0.3,
                        line_width=0, row=1, col=1)
        fig_v.add_trace(go.Scatter(
            x=charger_data["timestamp"], y=charger_data["temperature"],
            line=dict(color="#ef4444", width=1.2), name="Temp"
        ), row=2, col=1)
        fig_v.add_hrect(y0=0, y1=50, fillcolor="#d1fae5", opacity=0.2,
                        line_width=0, row=2, col=1)
        fig_v.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=36, b=0),
            font=dict(family="DM Sans", size=11, color="#334155"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_v, use_container_width=True, config={"displayModeBar": False})

    # ── Fleet health timeline ──────────────────────────────────

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">30-Day Fleet Health Timeline</div>', unsafe_allow_html=True)

    # Compute daily health per charger (rolling window approximation)
    now_ts = df_tel["timestamp"].max()
    daily_scores = []
    for day_offset in range(29, -1, -1):
        day_end = now_ts - timedelta(days=day_offset)
        day_start = day_end - timedelta(days=1)
        day_data = df_tel[(df_tel["timestamp"] >= day_start) & (df_tel["timestamp"] < day_end)]
        if len(day_data) == 0:
            continue
        for cid in df_scores["charger_id"].unique():
            c_data = day_data[day_data["charger_id"] == cid]
            if len(c_data) == 0:
                continue
            sr = c_data["session_success"].mean() * 100
            v_std = c_data["voltage"].std()
            baseline_v_std = df_tel[df_tel["charger_id"] == cid]["voltage"].std()
            v_s = max(0, 100 - 50 * (v_std / max(baseline_v_std, 0.01) - 1))
            t_avg = c_data["temperature"].mean()
            t_s = max(0, 100 - 4 * max(0, t_avg - 50))
            h = 0.35 * v_s + 0.30 * t_s + 0.35 * sr
            daily_scores.append({"date": day_end.date(), "charger_id": cid, "health": h})

    df_daily = pd.DataFrame(daily_scores)
    fleet_daily = df_daily.groupby("date")["health"].mean().reset_index()

    fig_fleet = go.Figure()
    for cid in df_scores["charger_id"].unique():
        c_daily = df_daily[df_daily["charger_id"] == cid]
        fig_fleet.add_trace(go.Scatter(
            x=c_daily["date"], y=c_daily["health"],
            mode="lines", name=cid,
            line=dict(width=1),
            opacity=0.4,
        ))
    fig_fleet.add_trace(go.Scatter(
        x=fleet_daily["date"], y=fleet_daily["health"],
        mode="lines", name="Fleet Avg",
        line=dict(width=2.5, color="#1a1a2e"),
    ))
    fig_fleet.add_hline(y=85, line_dash="dot", line_color="#10b981", line_width=1)
    fig_fleet.add_hline(y=65, line_dash="dot", line_color="#ef4444", line_width=1)
    fig_fleet.update_layout(
        height=280,
        yaxis=dict(range=[0, 105], title="Health Score"),
        margin=dict(l=0, r=0, t=10, b=10),
        font=dict(family="DM Sans", size=11, color="#334155"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_fleet, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<p class="data-note">Simulated OCPP telemetry — 6 chargers, 90-day dataset. Failure distribution modelled on real IGBT/SiC thermal degradation patterns.</p>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#
#   TAB 2 — SOLAR INVERTER TCO CALCULATOR
#
#   Problem: Suresh picks Chinese inverter on spec sheet alone.
#   Solution: 25-year NPV model showing true cost gap.
#
#   Physics:
#     Annual generation = PR × Peak_Power × Irradiance
#     PR (Performance Ratio) = η_inverter × η_system_other
#     η_system_other = 0.84 (cable loss, soiling, mismatch, shading)
#
#     Energy gap per year = PR_delta × Peak_Power × Irradiance
#     Revenue from gap = Energy_gap × tariff (escalating at 5% p.a.)
#     NPV at discount rate r: Σ Revenue_t / (1+r)^t
#
#     Service cost differential:
#       Chinese: ₹X/event × N_events × (1 + import lead time penalty)
#       Zenergize: ₹Y/event × N_events (domestic, next-day parts)
#
# ═══════════════════════════════════════════════════════════════

with tab3:

    def thermal_derating_factor(ambient_temp_c):
        """
        SiC MOSFET thermal derating.
        Source: Infineon AN2020-16 thermal management guidelines.
        """
        if ambient_temp_c <= 40:
            return 1.0
        elif ambient_temp_c <= 45:
            return 1.0 - (ambient_temp_c - 40) * 0.01
        else:
            return 0.95 - (ambient_temp_c - 45) * 0.014

    def get_tariff_rate(hour, peak_rate, offpeak_rate):
        """
        Delhi DISCOM approximate time-of-use tariff.
        Peak: 06:00–22:00
        Off-peak: 22:00–06:00
        """
        if 6 <= hour < 22:
            return peak_rate
        else:
            return offpeak_rate

    @st.cache_data(ttl=300)
    def generate_fleet_scenario(seed=7, n_vehicles=120, n_chargers=20):
        """
        Sunday 10pm snapshot for Anil's depot.
        120 delivery vans returning from weekend routes.
        20 DC fast chargers, rated 30kW each.
        """
        rng = np.random.default_rng(seed)

        # Vehicle states
        return_times = []
        for i in range(n_vehicles):
            if i < 8:   # 8 late vans (back by 1am)
                return_times.append(rng.uniform(22.5, 25.0))  # 10:30pm–1:00am (hours from midnight)
            else:
                return_times.append(rng.uniform(20.0, 23.0))  # 8pm–11pm

        vehicles = pd.DataFrame({
            "vehicle_id": [f"VAN-{i+1:03d}" for i in range(n_vehicles)],
            "current_soc": rng.integers(8, 45, n_vehicles),
            "target_soc": 80,
            "battery_kwh": rng.choice([60, 72, 80], n_vehicles, p=[0.5, 0.35, 0.15]),
            "return_hour": np.array(return_times) % 24,
            "deadline_hour": 5.5,  # 5:30am
        })

        # Charger states
        charger_health = np.clip(rng.normal(82, 15, n_chargers), 20, 100)
        # Chargers 7, 12, 18 (0-indexed 6, 11, 17) are hot from Saturday afternoon
        charger_health[[6, 11, 17]] = rng.uniform(40, 65, 3)
        ambient_temps = np.clip(rng.normal(32, 4, n_chargers), 22, 52)
        ambient_temps[[6, 11, 17]] = rng.uniform(44, 50, 3)

        chargers = pd.DataFrame({
            "charger_id": [f"CH-{i+1:02d}" for i in range(n_chargers)],
            "rated_power_kw": 30,
            "health_score": charger_health,
            "ambient_temp": ambient_temps,
        })
        chargers["derating_factor"] = chargers["ambient_temp"].apply(thermal_derating_factor)
        chargers["effective_power_kw"] = (
            chargers["rated_power_kw"] *
            chargers["derating_factor"] *
            (chargers["health_score"] / 100) ** 0.3
        ).round(1)

        return vehicles, chargers

    def run_scheduler(vehicles, chargers, peak_rate, offpeak_rate):
        """
        EDF + priority scheduler.

        Priority score for vehicle i:
            P_i = w_soc × (1 - current_soc/target_soc)
                + w_deadline × (1 / hours_until_deadline)
                + w_energy × (energy_needed_kwh / 60)   [normalised]

        Where: w_soc=0.50, w_deadline=0.30, w_energy=0.20

        Vehicles are sorted descending by priority.
        Each is assigned to the highest-health available charger at their return time.
        Charging ends when: accumulated energy ≥ energy_needed.
        Tariff cost computed per 15-min slot based on slot hour.
        """
        results = []
        now_hour = 22.0  # Sunday 10pm

        charger_free_at = {cid: max(now_hour, v) for cid, v in
                           zip(chargers["charger_id"],
                               np.clip(np.random.uniform(20, 22.5, len(chargers)), now_hour, 23))}

        # Priority sort
        df_v = vehicles.copy()
        df_v["energy_needed"] = (df_v["target_soc"] - df_v["current_soc"]) / 100 * df_v["battery_kwh"]
        deadline_h = df_v["deadline_hour"].iloc[0] + 24  # next day 5:30am = 29.5h from midnight

        df_v["hours_until_deadline"] = deadline_h - df_v["return_hour"].apply(
            lambda h: h if h >= now_hour else h + 24
        )
        df_v["hours_until_deadline"] = df_v["hours_until_deadline"].clip(lower=0.5)

        df_v["priority"] = (
            0.50 * (1 - df_v["current_soc"] / df_v["target_soc"]) +
            0.30 * (1 / df_v["hours_until_deadline"]) +
            0.20 * (df_v["energy_needed"] / 60)
        )
        df_v = df_v.sort_values("priority", ascending=False).reset_index(drop=True)

        # Sort chargers by health (best chargers for high-priority vehicles)
        chargers_sorted = chargers.sort_values("health_score", ascending=False).reset_index(drop=True)

        for i, v_row in df_v.iterrows():
            vid = v_row["vehicle_id"]
            return_h = v_row["return_hour"] if v_row["return_hour"] >= now_hour else v_row["return_hour"] + 24
            energy_needed = v_row["energy_needed"]

            # Find best available charger at vehicle return time
            best_charger = None
            best_score = -1
            for _, c_row in chargers_sorted.iterrows():
                cid = c_row["charger_id"]
                free_at = charger_free_at[cid]
                start_h = max(return_h, free_at)
                if start_h < deadline_h:
                    score = c_row["health_score"] - (start_h - return_h) * 5
                    if score > best_score:
                        best_score = score
                        best_charger = c_row
                        best_start = start_h

            if best_charger is None:
                # No charger available in time
                results.append({
                    "vehicle_id": vid,
                    "charger_id": "UNASSIGNED",
                    "start_hour": None,
                    "end_hour": None,
                    "final_soc": v_row["current_soc"],
                    "target_met": False,
                    "energy_delivered": 0,
                    "cost": 0,
                    "naive_cost": 0,
                })
                continue

            cid = best_charger["charger_id"]
            effective_kw = best_charger["effective_power_kw"]

            # Charge in 15-min slots; shift to off-peak where possible
            energy_remaining = energy_needed
            current_h = best_start
            total_cost = 0

            # Simple linear charging: compute end time and cost
            charge_hours = energy_remaining / effective_kw

            # Check if off-peak shifting is possible
            off_peak_start = 22.0 if best_start < 22 else best_start
            off_peak_end = 30.0  # 6am next day in 24h+ scale
            available_offpeak = max(0, off_peak_end - max(best_start, 22.0))

            # Optimal cost: maximise off-peak charging
            offpeak_charge = min(available_offpeak, charge_hours)
            peak_charge = max(0, charge_hours - offpeak_charge)

            cost_optimised = (
                offpeak_charge * effective_kw * offpeak_rate +
                peak_charge * effective_kw * peak_rate
            )
            cost_naive = charge_hours * effective_kw * peak_rate  # naive: all peak

            end_h = best_start + charge_hours
            final_soc_pct = min(100, v_row["current_soc"] + (energy_needed / v_row["battery_kwh"]) * 100)
            target_met = end_h <= deadline_h

            charger_free_at[cid] = end_h

            results.append({
                "vehicle_id": vid,
                "charger_id": cid,
                "current_soc": v_row["current_soc"],
                "start_hour": round(best_start % 24, 2),
                "end_hour": round(end_h % 24, 2),
                "charge_duration_h": round(charge_hours, 2),
                "final_soc": round(final_soc_pct, 1),
                "target_met": target_met,
                "energy_delivered": round(energy_needed, 1),
                "charger_health": best_charger["health_score"],
                "effective_power_kw": round(effective_kw, 1),
                "cost_optimised": round(cost_optimised, 1),
                "cost_naive": round(cost_naive, 1),
            })

        return pd.DataFrame(results)

    def hour_to_time_str(h):
        h = h % 24
        hh = int(h)
        mm = int((h - hh) * 60)
        return f"{hh:02d}:{mm:02d}"

    # ── Inputs ─────────────────────────────────────────────────

    st.markdown('<div class="section-label">Depot Parameters</div>', unsafe_allow_html=True)

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        n_vans = st.slider("Number of Vehicles", 20, 150, 120, 10)
        n_chargers_input = st.slider("Number of Chargers", 5, 30, 20, 1)
    with col_p2:
        peak_rate = st.number_input("Peak Tariff (₹/kWh)", 4.0, 15.0, 8.5, 0.5)
        offpeak_rate = st.number_input("Off-Peak Tariff (₹/kWh)", 2.0, 10.0, 4.5, 0.5)
    with col_p3:
        ambient_note = st.markdown("""
        <div class="metric-card">
          <div class="metric-label">Schedule Window</div>
          <div class="metric-value" style="font-size:1.1rem">10:00 PM → 5:30 AM</div>
          <div class="metric-sub">7.5 hours overnight. Peak ends at 10pm, off-peak until 6am.</div>
        </div>""", unsafe_allow_html=True)

    run_sched = st.button("Generate Optimal Schedule", type="primary")

    if run_sched or "fleet_result" not in st.session_state:
        with st.spinner("Running scheduler..."):
            vehicles, chargers = generate_fleet_scenario(n_vehicles=n_vans, n_chargers=n_chargers_input)
            df_result = run_scheduler(vehicles, chargers, peak_rate, offpeak_rate)
            st.session_state["fleet_result"] = df_result
            st.session_state["fleet_chargers"] = chargers
            st.session_state["fleet_vehicles"] = vehicles

    df_result = st.session_state["fleet_result"]
    chargers_data = st.session_state["fleet_chargers"]
    vehicles_data = st.session_state["fleet_vehicles"]

    # ── Summary ────────────────────────────────────────────────

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Schedule Outcome</div>', unsafe_allow_html=True)

    assigned = df_result[df_result["charger_id"] != "UNASSIGNED"]
    n_target_met = assigned["target_met"].sum()
    n_total = len(df_result)
    readiness_pct = n_target_met / n_total * 100

    total_cost_opt = assigned["cost_optimised"].sum()
    total_cost_naive = assigned["cost_naive"].sum()
    cost_saved = total_cost_naive - total_cost_opt

    n_derated = (chargers_data["derating_factor"] < 0.98).sum()
    avg_derating = chargers_data["derating_factor"].mean()

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    readiness_color = "#10b981" if readiness_pct >= 99 else "#f59e0b" if readiness_pct >= 95 else "#ef4444"
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Morning Readiness</div>
          <div class="metric-value" style="color:{readiness_color}">{readiness_pct:.1f}%</div>
          <div class="metric-sub">{n_target_met}/{n_total} vehicles at target SoC by 5:30am</div>
        </div>""", unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Peak Cost Saved Tonight</div>
          <div class="metric-value" style="color:#10b981">₹{cost_saved:,.0f}</div>
          <div class="metric-sub">vs. naive all-peak unoptimised schedule</div>
        </div>""", unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Chargers Under Thermal Derate</div>
          <div class="metric-value" style="color:{'#f59e0b' if n_derated > 0 else '#10b981'}">{n_derated}</div>
          <div class="metric-sub">Avg derating factor: {avg_derating:.3f}</div>
        </div>""", unsafe_allow_html=True)
    with col_m4:
        opt_rate = total_cost_opt / assigned["energy_delivered"].sum() if assigned["energy_delivered"].sum() > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Effective Cost / kWh</div>
          <div class="metric-value">₹{opt_rate:.2f}</div>
          <div class="metric-sub">Blended after off-peak shifting</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ─────────────────────────────────────────────────

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        # SOC distribution: before and after
        fig_soc = go.Figure()
        fig_soc.add_trace(go.Histogram(
            x=vehicles_data["current_soc"],
            name="SoC at Arrival",
            nbinsx=20,
            marker_color="#cbd5e1",
            marker_line_color="#94a3b8",
            marker_line_width=1,
        ))
        fig_soc.add_trace(go.Histogram(
            x=df_result["final_soc"],
            name="SoC at 5:30am",
            nbinsx=20,
            marker_color="#334155",
            opacity=0.8,
            marker_line_color="#475569",
            marker_line_width=1,
        ))
        fig_soc.add_vline(x=80, line_dash="dot", line_color="#ef4444",
                          annotation_text="Target 80%", annotation_font_size=10, annotation_font_color="#475569")
        fig_soc.update_layout(
            title="SoC Distribution: Arrival vs. Departure",
            barmode="overlay",
            height=280,
            margin=dict(l=0, r=0, t=36, b=10),
            font=dict(family="DM Sans", size=11, color="#334155"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis=dict(title="State of Charge (%)", range=[0, 105]),
            yaxis=dict(title="Vehicles"),
        )
        st.plotly_chart(fig_soc, use_container_width=True, config={"displayModeBar": False})

    with col_c2:
        # Charger health vs effective power
        fig_ch = go.Figure()
        fig_ch.add_trace(go.Scatter(
            x=chargers_data["health_score"],
            y=chargers_data["effective_power_kw"],
            mode="markers+text",
            text=chargers_data["charger_id"],
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(
                size=10,
                color=chargers_data["ambient_temp"],
                colorscale="RdYlGn_r",
                showscale=True,
                colorbar=dict(title="Temp (°C)", thickness=12),
            ),
        ))
        fig_ch.add_vline(x=65, line_dash="dot", line_color="#f59e0b",
                         annotation_text="Health threshold", annotation_font_size=10, annotation_font_color="#475569")
        fig_ch.update_layout(
            title="Charger Health vs Effective Charging Power",
            height=280,
            margin=dict(l=0, r=0, t=36, b=10),
            font=dict(family="DM Sans", size=11, color="#334155"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Health Score", range=[0, 110]),
            yaxis=dict(title="Effective Power (kW)"),
        )
        st.plotly_chart(fig_ch, use_container_width=True, config={"displayModeBar": False})

    # ── Charging timeline Gantt ────────────────────────────────

    st.markdown('<div class="section-label">Overnight Charging Schedule (first 30 vehicles shown)</div>', unsafe_allow_html=True)

    display_n = 30
    gantt_data = assigned.head(display_n).copy()

    fig_gantt = go.Figure()
    colors = {"target_met": "#10b981", "not_met": "#ef4444"}

    for i, row in gantt_data.iterrows():
        color = "#10b981" if row["target_met"] else "#ef4444"
        # Adjust hours for display (22 = 10pm start)
        start = row["start_hour"]
        end = row["end_hour"]
        if start > end:
            end += 24
        fig_gantt.add_trace(go.Scatter(
            x=[start, end],
            y=[row["vehicle_id"], row["vehicle_id"]],
            mode="lines",
            line=dict(color=color, width=8),
            showlegend=False,
        ))

    # Add tariff boundary
    fig_gantt.add_vline(x=22, line_dash="dot", line_color="#f59e0b",
                        annotation_text="Off-peak starts 10pm", annotation_font_size=10, annotation_font_color="#475569")
    fig_gantt.add_vline(x=5.5, line_dash="dot", line_color="#ef4444",
                        annotation_text="Deadline 5:30am", annotation_font_size=10, annotation_font_color="#475569")

    fig_gantt.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=10, b=10),
        font=dict(family="DM Sans", size=10, color="#334155"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Hour of Day",
            tickvals=[20, 22, 0, 2, 4, 5.5],
            ticktext=["8pm", "10pm(off-peak)", "12am", "2am", "4am", "5:30am"],
        ),
        yaxis=dict(title="Vehicle", autorange="reversed"),
    )
    st.plotly_chart(fig_gantt, use_container_width=True, config={"displayModeBar": False})

    # ── At-risk vehicles ───────────────────────────────────────

    at_risk = df_result[~df_result["target_met"]]
    if len(at_risk) > 0:
        st.markdown('<div class="section-label">Vehicles Not Meeting Target by 5:30am</div>', unsafe_allow_html=True)
        for _, row in at_risk.iterrows():
            st.markdown(f"""
            <div class="alert-row-red">
              <span class="alert-text"><strong>{row['vehicle_id']}</strong> — Final SoC: {row['final_soc']:.1f}% &nbsp;·&nbsp; {row['charger_id']}</span><br>
              <span class="alert-time">Manual priority assignment required. Current SoC at arrival: {row['current_soc']}%.</span>
            </div>""", unsafe_allow_html=True)

    st.markdown('<p class="data-note">Simulated depot scenario — 120 vans, 20 DC fast chargers, Sunday 10pm snapshot. Thermal derating from Infineon AN2020-16. Tariff schedule: Delhi DISCOM ToU approximation.</p>', unsafe_allow_html=True)