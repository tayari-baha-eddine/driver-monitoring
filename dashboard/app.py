"""Real-time Driver Monitoring dashboard with Risk Engine visualization."""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# Config
# ============================================================

DB_PATH = Path("data/driver_events.db")

STATE_COLORS = {
    "ALERT":      "#00c800",
    "DROWSY":     "#ff8c00",
    "DISTRACTED": "#ffc800",
    "CRITICAL":   "#ff0000",
    "NO_DRIVER":  "#888888",
}

STATE_ICONS = {
    "ALERT":      "✅",
    "DROWSY":     "😴",
    "DISTRACTED": "👀",
    "CRITICAL":   "🚨",
    "NO_DRIVER":  "❓",
}

st.set_page_config(
    page_title="Driver Monitoring",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
    .state-banner {
        font-size: 2.4rem; font-weight: 800; text-align: center;
        padding: 1.2rem; border-radius: 14px; margin-bottom: 0.5rem;
        letter-spacing: 2px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 0 currentColor; }
        50%      { box-shadow: 0 0 20px 2px currentColor; }
    }
    .alert-item {
        padding: 0.6rem 0.9rem; border-radius: 8px; margin-bottom: 0.5rem;
        border-left: 4px solid; background-color: #1e1e1e;
    }
    .alert-time { font-size: 0.75rem; color: #999; }
    .alert-reason { font-size: 0.9rem; color: #ddd; margin-top: 0.2rem; }
    .alert-state { font-weight: 700; }
    .driver-badge {
        display: inline-block; padding: 0.2rem 0.6rem;
        border-radius: 20px; background-color: #0096ff22;
        color: #0096ff; font-weight: 700; font-size: 0.85rem;
        border: 1px solid #0096ff;
    }
    .pending-badge {
        display: inline-block; padding: 0.2rem 0.6rem;
        border-radius: 20px; background-color: #ffc80022;
        color: #ffc800; font-weight: 700; font-size: 0.8rem;
        border: 1px solid #ffc800; margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Data
# ============================================================

@st.cache_resource
def get_db_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)


def load_data(limit: int) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()

    con = get_db_connection()
    try:
        df = pd.read_sql(
            f"SELECT * FROM events ORDER BY ts DESC LIMIT {limit}", con
        )
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["ts"], unit="s")
    return df.sort_values("datetime").reset_index(drop=True)


def apply_filters(df: pd.DataFrame, states: list[str],
                  time_window_min: int | None,
                  drivers: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    if states:
        df = df[df["state"].isin(states)]
    if drivers and "driver" in df.columns:
        df = df[df["driver"].isin(drivers)]
    if time_window_min is not None:
        cutoff = df["datetime"].max() - timedelta(minutes=time_window_min)
        df = df[df["datetime"] >= cutoff]
    return df.reset_index(drop=True)


# ============================================================
# Components
# ============================================================

def render_state_banner(latest: pd.Series) -> None:
    state = latest["state"]
    color = STATE_COLORS.get(state, "#ffffff")
    icon = STATE_ICONS.get(state, "")
    reason = latest.get("reason", "") or ""

    if not reason:
        if state == "ALERT":
            reason = "Conduite normale · Tous les indicateurs sont bons"
        elif state == "NO_DRIVER":
            reason = "Aucun conducteur détecté devant la caméra"
        else:
            reason = state

    attention = latest.get("attention", 0.0)
    vigilance = latest.get("vigilance", 0.0)
    fps = latest.get("fps", 0.0)
    timestamp = latest["datetime"].strftime("%H:%M:%S")
    driver = latest.get("driver", "unknown") or "unknown"
    driver_txt = (f'<span class="driver-badge">👤 {driver}</span>'
                  if driver and driver != "unknown" else "")

    st.markdown(
        f"""
        <div class="state-banner"
             style="background-color:{color}22; color:{color};
                    border:2px solid {color};">
            {icon} {state}
        </div>
        <p style="text-align:center; color:#ddd; margin-top:-0.5rem;
                  font-size:1.05rem;">
            {reason}
        </p>
        <p style="text-align:center; color:#888; font-size:0.85rem;
                  margin-top:-0.3rem;">
            Attention <b style="color:#00c800;">{attention:.0f}%</b> &nbsp;·&nbsp;
            Vigilance <b style="color:#0096ff;">{vigilance:.0f}%</b> &nbsp;·&nbsp;
            FPS <b>{fps:.1f}</b> &nbsp;·&nbsp;
            {timestamp}
            {(" &nbsp;·&nbsp; " + driver_txt) if driver_txt else ""}
        </p>
        """,
        unsafe_allow_html=True,
    )


def render_gauge(value: float, title: str, key: str,
                 invert: bool = False) -> go.Figure:
    """Circular gauge. If invert=True, high value is bad."""
    if invert:
        if value >= 70:
            bar_color = "#ff0000"
        elif value >= 45:
            bar_color = "#ff8c00"
        elif value >= 20:
            bar_color = "#ffc800"
        else:
            bar_color = "#00c800"
    else:
        if value >= 75:
            bar_color = "#00c800"
        elif value >= 50:
            bar_color = "#ffc800"
        elif value >= 25:
            bar_color = "#ff8c00"
        else:
            bar_color = "#ff0000"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"size": 34}},
        title={"text": title, "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#666"},
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "#1e1e1e",
            "borderwidth": 1, "bordercolor": "#333",
            "steps": [
                {"range": [0, 25],   "color": "rgba(255,0,0,0.12)"},
                {"range": [25, 50],  "color": "rgba(255,140,0,0.12)"},
                {"range": [50, 75],  "color": "rgba(255,200,0,0.12)"},
                {"range": [75, 100], "color": "rgba(0,200,0,0.12)"},
            ],
        },
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=15, r=15, t=45, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#fff"},
    )
    return fig


def render_alerts_panel(df: pd.DataFrame, n: int = 5) -> None:
    alerts = df[df["state"] != "ALERT"].tail(n).iloc[::-1]
    if alerts.empty:
        st.info("🎉 Aucune alerte récente.")
        return

    for _, row in alerts.iterrows():
        state = row["state"]
        color = STATE_COLORS.get(state, "#fff")
        icon = STATE_ICONS.get(state, "")
        ts = row["datetime"].strftime("%H:%M:%S")
        reason = row.get("reason", "") or state
        attention = row.get("attention", 0.0)
        driver = row.get("driver", "")
        driver_str = f" · 👤 {driver}" if driver and driver != "unknown" else ""
        risk = row.get("risk_smoothed", 0.0)

        st.markdown(
            f"""
            <div class="alert-item" style="border-left-color:{color};">
                <div class="alert-time">{ts}{driver_str} · Risk {risk:.0f}</div>
                <div class="alert-state" style="color:{color};">
                    {icon} {state} &nbsp;·&nbsp;
                    <span style="color:#999;">Attention {attention:.0f}%</span>
                </div>
                <div class="alert-reason">{reason}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_risk_chart(df: pd.DataFrame) -> None:
    """Show raw vs smoothed risk over time — this proves the smoothing works."""
    if "risk_smoothed" not in df.columns:
        st.info("Colonne risk indisponible (base ancienne).")
        return

    fig = go.Figure()
    if "risk_instant" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=df["risk_instant"],
            name="Risk instantané", line=dict(color="#666", width=1, dash="dot"),
            opacity=0.6,
        ))
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["risk_smoothed"],
        name="Risk lissé (EMA)", line=dict(color="#ff3366", width=2.5),
        fill="tozeroy", fillcolor="rgba(255,51,102,0.1)",
    ))
    fig.update_layout(
        title="🧠 Risk Engine : brut vs lissé",
        yaxis=dict(range=[0, 105], title="Risk score"),
        xaxis_title="Temps",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_scores_chart(df: pd.DataFrame) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["attention"],
        name="Attention", line=dict(color="#00c800", width=2),
        fill="tozeroy", fillcolor="rgba(0,200,0,0.1)",
    ))
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["vigilance"],
        name="Vigilance", line=dict(color="#0096ff", width=2),
    ))
    fig.update_layout(
        title="Scores en temps réel",
        yaxis=dict(range=[0, 105]),
        xaxis_title="Temps",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_state_pie(df: pd.DataFrame) -> None:
    counts = df["state"].value_counts().reset_index()
    counts.columns = ["state", "count"]
    fig = px.pie(
        counts, names="state", values="count",
        title="Répartition des états",
        color="state", color_discrete_map=STATE_COLORS, hole=0.45,
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_stats(df: pd.DataFrame) -> None:
    total = len(df)
    phone_count = int(df["phone"].sum())
    avg_fps = df["fps"].mean()
    avg_lat = df["latency_ms"].mean()
    duration_min = (df["ts"].max() - df["ts"].min()) / 60.0 if total > 1 else 0.0
    transitions = int((df["state"] != df["state"].shift()).sum()) - 1

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Événements", total)
    c2.metric("Téléphone", phone_count)
    c3.metric("FPS moyen", f"{avg_fps:.1f}")
    c4.metric("Latence", f"{avg_lat:.0f} ms")
    c5.metric("Transitions", max(0, transitions))


def render_event_table(df: pd.DataFrame) -> None:
    cols = ["datetime", "driver", "state", "reason", "attention",
            "vigilance", "risk_smoothed", "drowsiness", "distraction",
            "ear", "yaw", "phone", "fps"]
    existing = [c for c in cols if c in df.columns]
    display = df.tail(30)[existing].iloc[::-1].copy()
    rename = {"datetime": "Heure", "driver": "Conducteur", "state": "État",
              "reason": "Raison", "attention": "Attention",
              "vigilance": "Vigilance", "risk_smoothed": "Risk",
              "drowsiness": "Somnolence", "distraction": "Distraction",
              "ear": "EAR", "yaw": "Yaw", "phone": "Tél.", "fps": "FPS"}
    display = display.rename(columns=rename)

    st.dataframe(
        display, use_container_width=True, hide_index=True,
        column_config={
            "Attention": st.column_config.ProgressColumn(
                "Attention", format="%.0f%%", min_value=0, max_value=100),
            "Vigilance": st.column_config.ProgressColumn(
                "Vigilance", format="%.0f%%", min_value=0, max_value=100),
            "Risk": st.column_config.ProgressColumn(
                "Risk", format="%.0f", min_value=0, max_value=100),
            "Somnolence": st.column_config.ProgressColumn(
                "Somnolence", format="%.0f", min_value=0, max_value=100),
            "Distraction": st.column_config.ProgressColumn(
                "Distraction", format="%.0f", min_value=0, max_value=100),
        },
    )


# ============================================================
# Sidebar
# ============================================================

def render_sidebar() -> dict:
    with st.sidebar:
        st.header("⚙️ Paramètres")
        refresh = st.slider("Rafraîchissement (s)", 1, 10, 2)
        limit = st.slider("Événements chargés", 100, 5000, 1000, step=100)

        st.divider()
        st.subheader("🔍 Filtres")

        states = st.multiselect(
            "États à afficher",
            options=list(STATE_COLORS.keys()),
            default=list(STATE_COLORS.keys()),
        )

        drivers: list[str] = []
        if DB_PATH.exists():
            try:
                con = get_db_connection()
                rows = con.execute(
                    "SELECT DISTINCT driver FROM events WHERE driver IS NOT NULL"
                ).fetchall()
                drivers = [r[0] for r in rows if r[0]]
            except Exception:
                drivers = []

        selected_drivers: list[str] = []
        if drivers:
            selected_drivers = st.multiselect(
                "Conducteurs", options=drivers, default=drivers)

        time_options = {
            "Toutes": None,
            "5 dernières minutes": 5,
            "15 dernières minutes": 15,
            "30 dernières minutes": 30,
            "1 heure": 60,
        }
        time_label = st.selectbox("Période", list(time_options.keys()), index=0)
        time_window = time_options[time_label]

        st.divider()
        auto_refresh = st.toggle("Auto-refresh", value=True)
        if st.button("🔄 Rafraîchir maintenant"):
            st.rerun()

        st.divider()
        st.caption("💾 Base :")
        st.code(str(DB_PATH), language=None)

        return {
            "refresh": refresh,
            "limit": limit,
            "states": states,
            "drivers": selected_drivers,
            "time_window": time_window,
            "auto_refresh": auto_refresh,
        }


# ============================================================
# Main
# ============================================================

def main() -> None:
    st.title("🚗 Driver Monitoring — Live Dashboard")

    cfg = render_sidebar()
    placeholder = st.empty()

    while True:
        raw_df = load_data(cfg["limit"])
        df = apply_filters(raw_df, cfg["states"], cfg["time_window"],
                           cfg["drivers"])

        with placeholder.container():
            if raw_df.empty:
                st.warning("📭 Aucun événement. Lance `python main.py`.")
            elif df.empty:
                st.info("🔍 Aucun événement ne correspond aux filtres.")
            else:
                latest = df.iloc[-1]

                col_state, col_alerts = st.columns([2, 1])
                with col_state:
                    render_state_banner(latest)
                with col_alerts:
                    st.subheader("🔔 Alertes récentes")
                    render_alerts_panel(df, n=5)

                st.divider()
                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    st.plotly_chart(
                        render_gauge(latest["attention"], "Attention", "att"),
                        use_container_width=True)
                with c2:
                    st.plotly_chart(
                        render_gauge(latest["vigilance"], "Vigilance", "vig"),
                        use_container_width=True)
                with c3:
                    st.plotly_chart(
                        render_gauge(latest.get("drowsiness", 0.0),
                                     "Somnolence", "drow", invert=True),
                        use_container_width=True)
                with c4:
                    risk_val = latest.get("risk_smoothed", 0.0)
                    st.plotly_chart(
                        render_gauge(risk_val, "🧠 Risk Engine", "risk",
                                     invert=True),
                        use_container_width=True)

                st.divider()
                render_risk_chart(df)

                st.divider()
                c4, c5 = st.columns(2)
                with c4:
                    render_scores_chart(df)
                with c5:
                    render_state_pie(df)

                st.divider()
                st.subheader("📈 Statistiques de session")
                render_stats(df)

                st.divider()
                st.subheader("📋 Derniers événements")
                render_event_table(df)

                st.caption(
                    f"Dernière mise à jour : "
                    f"{latest['datetime'].strftime('%H:%M:%S')}  |  "
                    f"Affichés : {len(df)} / {len(raw_df)} événements"
                )

        if not cfg["auto_refresh"]:
            break

        time.sleep(cfg["refresh"])
        st.rerun()


if __name__ == "__main__":
    main()