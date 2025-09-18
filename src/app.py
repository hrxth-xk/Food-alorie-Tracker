# src/app.py
"""
Food Calorie Tracker — auto-refresh on add.
Overwrite src/app.py with this file.
"""

import os
import io
import math
import re
import hashlib
from datetime import date, timedelta
from typing import List, Dict, Any, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# ---- load .env from project root (one level up from src) ----
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dotenv_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path)

# ---- internal modules ----
from api_client import search, load_local_master
from db import init_db, add_meal, get_logs_for_date, get_logs_between

# optional PDF libs
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False



# init DB
init_db()

# ---------------- Dedup helpers ----------------
_normalize_re = re.compile(r"[^\w\s]", flags=re.UNICODE)
_parenthetical_re = re.compile(r"\(.*?\)")

def _normalize_name(name: str) -> str:
    if not name:
        return ""
    s = str(name).lower()
    s = _parenthetical_re.sub("", s)
    s = _normalize_re.sub(" ", s)
    s = " ".join(s.split())
    return s.strip()

def _completeness_score(item: Dict) -> int:
    score = 0
    for k in ("calories", "protein", "carbs", "fat"):
        try:
            if item.get(k) is not None and str(item.get(k)).strip() != "":
                score += 1
        except Exception:
            pass
    if item.get("fdcId") or item.get("id") or item.get("local_id"):
        score += 1
    src = (item.get("source") or "").lower()
    if src == "local":
        score += 1
    return score

def dedupe_results(results: List[Dict], keep_all_collapsed: bool = False) -> List[Dict]:
    if not results:
        return results
    groups = {}
    order = []
    for item in results:
        name = item.get("name") or item.get("description") or item.get("food") or ""
        key = _normalize_name(name)
        if not key:
            key = f"__id_{item.get('fdcId') or item.get('id') or hash(name)}"
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(item)
    deduped = []
    for key in order:
        group = groups[key]
        if len(group) == 1:
            item = dict(group[0])
            if keep_all_collapsed:
                item["_collapsed"] = []
            deduped.append(item)
            continue
        scored = []
        for it in group:
            sc = _completeness_score(it)
            name_len = len(str(it.get("name") or ""))
            scored.append((sc, -name_len, it))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = dict(scored[0][2])
    if keep_all_collapsed:
        collapsed = [ { "name": i.get("name"), "source": i.get("source"), "calories": i.get("calories") } for (_,_,i) in scored[1:] ]
        best["_collapsed"] = collapsed
    deduped.append(best)
    return deduped

# ---------------- Helpers ----------------
def fmt_kcal(n):
    try:
        return f"{int(n):,} kcal"
    except Exception:
        return "N/A"

def df_from_meal_logs(rows):
    normalized = []
    for r in (rows or []):
        if isinstance(r, dict) or hasattr(r, "get"):
            date_val = r.get("log_date") or r.get("date") or r.get("created_at")
            food_val = r.get("food_name") or r.get("food")
            calories = r.get("calories") if r.get("calories") is not None else r.get("kcal") if r.get("kcal") is not None else 0
            protein = r.get("protein") if r.get("protein") is not None else 0
            carbs = r.get("carbs") if r.get("carbs") is not None else 0
            fat = r.get("fat") if r.get("fat") is not None else 0
            serving = r.get("serving") or ""
        else:
            date_val = getattr(r, "log_date", getattr(r, "date", None))
            food_val = getattr(r, "food_name", getattr(r, "food", None))
            calories = getattr(r, "calories", 0)
            protein = getattr(r, "protein", 0)
            carbs = getattr(r, "carbs", 0)
            fat = getattr(r, "fat", 0)
            serving = getattr(r, "serving", "")
        try:
            calories = float(calories) if calories is not None and calories != "" else 0.0
        except Exception:
            calories = 0.0
        try:
            protein = float(protein) if protein is not None and protein != "" else 0.0
        except Exception:
            protein = 0.0
        try:
            carbs = float(carbs) if carbs is not None and carbs != "" else 0.0
        except Exception:
            carbs = 0.0
        try:
            fat = float(fat) if fat is not None and fat != "" else 0.0
        except Exception:
            fat = 0.0
        normalized.append({
            "date": date_val,
            "food": food_val,
            "serving": serving,
            "calories": calories,
            "protein": protein,
            "carbs": carbs,
            "fat": fat,
        })
    if not normalized:
        return pd.DataFrame(columns=["date", "food", "serving", "calories", "protein", "carbs", "fat"])
    df = pd.DataFrame(normalized)
    try:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    except Exception:
        pass
    return df

def calc_macro_sums(df):
    if df is None or df.empty:
        return {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
    return {
        "calories": float(df["calories"].sum()) if "calories" in df.columns else 0.0,
        "protein": float(df["protein"].sum()) if "protein" in df.columns else 0.0,
        "carbs": float(df["carbs"].sum()) if "carbs" in df.columns else 0.0,
        "fat": float(df["fat"].sum()) if "fat" in df.columns else 0.0,
    }

def compute_macro_targets(goal_kcal, prot_pct, carbs_pct, fat_pct):
    prot_kcal = goal_kcal * (prot_pct / 100.0)
    carbs_kcal = goal_kcal * (carbs_pct / 100.0)
    fat_kcal = goal_kcal * (fat_pct / 100.0)
    return {
        "protein_g": prot_kcal / 4.0 if prot_kcal else 0.0,
        "carbs_g": carbs_kcal / 4.0 if carbs_kcal else 0.0,
        "fat_g": fat_kcal / 9.0 if fat_kcal else 0.0,
    }

# ---------------- Gauge ----------------
def rotated_filler_gauge(total_kcal, goal_kcal, width=920, height=420, pie_rotation=270):
    import math
    if goal_kcal is None or goal_kcal <= 0:
        goal_kcal = 2000
    total = float(total_kcal or 0.0)
    DISPLAY_OVERSHOOT = 1.2
    display_max = goal_kcal * DISPLAY_OVERSHOOT

    zones = [
        (0.0, 0.5, "#dff6ec"),
        (0.5, 0.85, "#fff6e6"),
        (0.85, 1.0, "#fff0f0"),
        (1.0, DISPLAY_OVERSHOOT, "#ffb9b9"),
    ]

    arc_values = []
    arc_colors = []
    for s, e, col in zones:
        start_k = s * goal_kcal
        end_k = min(e * goal_kcal, display_max)
        val = max(0.0, end_k - start_k)
        if val > 0:
            arc_values.append(val)
            arc_colors.append(col)
    arc_values.append(display_max)
    arc_colors.append("rgba(0,0,0,0)")

    fig = go.Figure()
    PIE_ROTATION = pie_rotation

    fig.add_trace(go.Pie(
        values=arc_values,
        rotation=PIE_ROTATION,
        hole=0.62,
        sort=False,
        direction='clockwise',
        marker=dict(colors=arc_colors, line=dict(color='#0f1113', width=2)),
        textinfo='none',
        showlegend=False
    ))

    progress = min(total, display_max)
    remainder = max(display_max - progress, 0.0)
    inner_values = [progress, remainder, display_max]
    inner_colors = ["#39d08a", "rgba(255,255,255,0.03)", "rgba(0,0,0,0)"]
    fig.add_trace(go.Pie(
        values=inner_values,
        rotation=PIE_ROTATION,
        hole=0.78,
        sort=False,
        direction='clockwise',
        marker=dict(colors=inner_colors, line=dict(color='rgba(0,0,0,0)', width=0)),
        textinfo='none',
        showlegend=False
    ))

    goal_ratio = min(goal_kcal / display_max, 1.0)
    goal_angle_deg = 180 * (1 - goal_ratio)
    g_angle = goal_angle_deg + PIE_ROTATION
    gtheta = math.radians(g_angle)
    cx, cy = 0.5, 0.5
    inner_r = 0.58
    outer_r = 0.66
    gx1 = cx + inner_r * math.cos(gtheta)
    gy1 = cy + inner_r * math.sin(gtheta)
    gx2 = cx + outer_r * math.cos(gtheta)
    gy2 = cy + outer_r * math.sin(gtheta)
    fig.add_shape(type="line",
                  x0=gx1, y0=gy1, x1=gx2, y1=gy2,
                  line=dict(color="#ff3b3b", width=5),
                  xref='paper', yref='paper')

    tick_fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
    r_base = 0.52
    for frac in tick_fracs:
        frac_display = frac * (goal_kcal / display_max)
        angle_deg = 180 * (1 - frac_display)
        angle_deg_rot = angle_deg + PIE_ROTATION
        rad = math.radians(angle_deg_rot)
        tx = cx + (r_base + 0.13) * math.cos(rad)
        ty = cy + (r_base + 0.13) * math.sin(rad)
        lbl = int(goal_kcal * frac)
        fig.add_annotation(x=tx, y=ty, xref='paper', yref='paper',
                           text=str(lbl), showarrow=False,
                           font=dict(size=12, color='#d6d7db'))

    remaining = max(goal_kcal - total, 0)
    fig.add_annotation(x=0.5, y=0.48, xref='paper', yref='paper',
                       text=f"<span style='font-size:44px;color:#e6e9ee'><b>{int(total):,} kcal</b></span>",
                       showarrow=False)
    fig.add_annotation(x=0.5, y=0.34, xref='paper', yref='paper',
                       text=f"<span style='color:#9aa0a6'>Remaining</span><br><b style='font-size:13px;color:#bfc4c8'>{int(remaining):,} kcal</b>",
                       showarrow=False)

    fig.update_layout(margin=dict(l=6, r=6, t=6, b=6),
                      height=height, width=width,
                      paper_bgcolor="#0f1113", plot_bgcolor="#0f1113",
                      showlegend=False)
    fig.update_traces(textinfo='none', hoverinfo='none')
    return fig

# ---------------- macros chart ----------------
def plot_macros_consumed_vs_target(consumed, targets, height=420):
    macros = ["Protein", "Carbs", "Fat"]
    consumed_vals = [consumed.get("protein", 0), consumed.get("carbs", 0), consumed.get("fat", 0)]
    target_vals = [targets.get("protein_g", 0), targets.get("carbs_g", 0), targets.get("fat_g", 0)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=macros, y=target_vals, name="Target (g)", marker_color="#9ec9ff"))
    fig.add_trace(go.Bar(x=macros, y=consumed_vals, name="Consumed (g)", marker_color="#2d6cdf"))
    fig.update_layout(barmode='group', height=height, paper_bgcolor="#0f1113", plot_bgcolor="#0f1113",
                      font=dict(color='#d6d7db'), legend=dict(bgcolor='rgba(0,0,0,0)'), margin=dict(t=30,b=20))
    return fig

# ---------------- add + refresh helper ----------------
def add_and_refresh(**kwargs):
    """
    Call add_meal(...) and if successful, show success and rerun the app
    so the new log appears immediately.
    """
    try:
        rowid = add_meal(**kwargs)
        if rowid:
            st.success(f"Added {kwargs.get('food_name')} — {fmt_kcal(kwargs.get('calories',0))}")
            # auto refresh UI to show new log
            st.rerun()
        else:
            st.error("Add did not return a row id. Check DB.")
    except Exception as e:
        st.error(f"Failed to add: {e}")
        with st.expander("Traceback", expanded=False):
            import traceback
            st.code(traceback.format_exc())

# ---------------- UI ----------------
st.markdown("<h1 style='color:#39d08a; font-size:42px;'>🍽️ BiteWise</h1>", unsafe_allow_html=True)
st.caption("Your personal calorie & macro tracker")

st.set_page_config(page_title="Food Calorie Tracker", layout="wide")
st.markdown("<h1 style='margin:8px 0px'>Today</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Profile & Goals")
with st.sidebar.expander("Profile (expand)", expanded=False):
    _name = st.text_input("Name", value=st.session_state.get("name", "You"), key="name")
    _age = st.number_input("Age", min_value=10, max_value=120, value=st.session_state.get("age", 30), key="age")
    _sex = st.selectbox("Sex", ["male","female","other"], index=0, key="sex")
    _weight = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=st.session_state.get("weight",70.0), step=0.5, key="weight")
    _height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=st.session_state.get("height",175.0), step=0.5, key="height")

st.sidebar.markdown("**Macro targets**")
c1, c2 = st.sidebar.columns(2)
protein_pct = c1.slider("Protein %", 0, 60, st.session_state.get("protein_pct", 20), key="protein_pct")
carbs_pct = c2.slider("Carbs %", 0, 80, st.session_state.get("carbs_pct", 50), key="carbs_pct")
fat_pct = st.sidebar.slider("Fat %", 0, 50, st.session_state.get("fat_pct", 30), key="fat_pct")

default_goal = int(st.session_state.get("goal_kcal", 2000))
goal_kcal = st.sidebar.number_input("Daily calorie goal (kcal)", min_value=800, max_value=8000, value=default_goal, step=50, key="goal_kcal")

st.sidebar.markdown("---")
if st.sidebar.checkbox("Show quick stats", value=True, key="show_quick_stats"):
    st.sidebar.subheader("Quick stats (7d)")
    end_q = date.today()
    start_q = end_q - timedelta(days=6)
    rows_q = get_logs_between(start_q, end_q)
    if rows_q:
        df_q = df_from_meal_logs(rows_q)
        daily = df_q.groupby("date")["calories"].sum().reset_index()
        st.sidebar.write(f"7d avg: {daily['calories'].mean():.0f} kcal")
    else:
        st.sidebar.write("No recent logs")
st.sidebar.markdown("---")
if st.sidebar.button("Jump to Today"):
    qp = dict(st.query_params)
    qp["tab"] = "today"
    # keep the tab param in session (we don't use experimental_set_query_params)
    st.session_state["_query_params"] = qp

# Tabs
tabs = st.tabs(["Search & Add", "Daily Log", "Progress", "Export"])
search_tab, daily_tab, progress_tab, export_tab = tabs

# Today's totals
logs_today = get_logs_for_date(date.today())
df_today = df_from_meal_logs(logs_today) if logs_today is not None else pd.DataFrame()
totals = calc_macro_sums(df_today)

# small CSS for compact spacing and separators
st.markdown(
    """
    <style>
    .compact-nutrients { margin-top:4px; margin-bottom:2px; color: #bfc4c8; font-size:14px; }
    .stButton>button { padding:6px 10px; font-size:13px; }
    hr.result-sep { margin:6px 0; border:0.5px solid #333; }
    </style>
    """,
    unsafe_allow_html=True,
)

# SEARCH & ADD tab
with search_tab:
    # top metric only
    st.markdown("<div style='display:flex;align-items:flex-start;gap:30px'>", unsafe_allow_html=True)
    left_block = st.columns(1)[0]
    with left_block:
        st.markdown("**Calories consumed**")
        st.markdown(f"<div style='font-size:64px;line-height:0.9;color:#e6e9ee;margin-top:8px'><b>{int(totals['calories']):,} kcal</b></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # graphs below
    gcol, mcol = st.columns([2, 1])
    with gcol:
        gauge_fig = rotated_filler_gauge(totals["calories"], goal_kcal, width=920, height=420)
        st.plotly_chart(gauge_fig, use_container_width=True)
    with mcol:
        targets = compute_macro_targets(goal_kcal, protein_pct, carbs_pct, fat_pct)
        figm = plot_macros_consumed_vs_target(totals, targets, height=420)
        st.plotly_chart(figm, use_container_width=True)
    

    st.header("Search and Add Food")
    q = st.text_input("Search for a food (try: banana, chkn, quinoa)", key="search_input")
    try:
        local_df = load_local_master()
        master_names = local_df["name"].astype(str).tolist() if not local_df.empty else []
    except Exception:
        local_df = pd.DataFrame()
        master_names = []

    suggestions = []
    if q and master_names:
        ql = q.strip().lower()
        suggestions = [n for n in master_names if ql in n.lower()][:6]

    if suggestions:
        st.markdown("**Suggestions (local)**")
        cols = st.columns(len(suggestions))
        for i, s in enumerate(suggestions):
            if cols[i].button(s, key=f"sugg_{i}_{hashlib.sha1(s.encode()).hexdigest()[:6]}"):
                st.session_state["search_input"] = s
                q = s

    if q:
        raw_results = search(q, use_api_first=True, limit=60)
        results = dedupe_results(raw_results, keep_all_collapsed=True)
        if results:
            st.markdown("### Results")
            for i, r in enumerate(results):
                name = r.get("name") or r.get("description") or "Unknown"
                kcal_db = r.get("calories")
                protein_db = r.get("protein")
                carbs_db = r.get("carbs")
                fat_db = r.get("fat")
                source = r.get("source") or "local"
                item_id = r.get("fdcId") or f"local_{hashlib.sha1(name.encode()).hexdigest()[:8]}"

                cols = st.columns([6, 2])
                left, right = cols[0], cols[1]

                with left:
                    st.markdown(f"<b style='font-size:18px'>{name}</b>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='compact-nutrients'>Source: {source} • {fmt_kcal(kcal_db)} • "
                        f"P {protein_db or '-'}g  C {carbs_db or '-'}g  F {fat_db or '-'}g</div>",
                        unsafe_allow_html=True
                    )
                    if r.get("_collapsed"):
                        with st.expander("Other variants", expanded=False):
                            for other in r["_collapsed"]:
                                st.write(f"- {other.get('name')} • {fmt_kcal(other.get('calories'))} • {other.get('source')}")

                with right:
                    if st.button("Quick Add", key=f"quick_{item_id}_{i}"):
                        # ALWAYS use add_and_refresh to ensure auto-reload after successful insert
                        add_and_refresh(
                            food_name=name,
                            calories=float(kcal_db or 0),
                            protein=protein_db,
                            carbs=carbs_db,
                            fat=fat_db,
                            serving="1 serving",
                            source=source,
                        )
                    with st.expander("Custom", expanded=False):
                        mode = st.radio("Mode", ("grams", "servings"), index=0, key=f"mode_{item_id}_{i}")
                        if mode == "grams":
                            default_c100 = kcal_db if kcal_db is not None else 100.0
                            c100 = st.number_input("kcal/100g", value=float(default_c100), step=1.0, key=f"c100_{item_id}_{i}")
                            grams = st.number_input("g", value=100.0, step=1.0, key=f"grams_{item_id}_{i}")
                            computed_kcal = c100 * (grams / 100.0)
                            p_val = protein_db * grams / 100.0 if protein_db is not None else None
                            c_val = carbs_db * grams / 100.0 if carbs_db is not None else None
                            f_val = fat_db * grams / 100.0 if fat_db is not None else None
                            st.write(f"{fmt_kcal(computed_kcal)} • P {p_val or '-'}g C {c_val or '-'}g F {f_val or '-'}g")
                            if st.button("Add", key=f"addgrams_{item_id}_{i}"):
                                add_and_refresh(
                                    food_name=name,
                                    calories=float(computed_kcal or 0),
                                    protein=p_val,
                                    carbs=c_val,
                                    fat=f_val,
                                    serving=f"{grams:.0f} g",
                                    source=source,
                                )
                        else:
                            servings = st.number_input("Servings", value=1.0, step=0.1, key=f"serv_{item_id}_{i}")
                            per_serv = float(kcal_db) if kcal_db is not None else st.number_input("kcal per serving", value=100.0, step=1.0, key=f"perserv_{item_id}_{i}")
                            computed_kcal = per_serv * servings
                            p_val = protein_db * servings if protein_db is not None else None
                            c_val = carbs_db * servings if carbs_db is not None else None
                            f_val = fat_db * servings if fat_db is not None else None
                            st.write(f"{fmt_kcal(computed_kcal)} • P {p_val or '-'}g C {c_val or '-'}g F {f_val or '-'}g")
                            if st.button("Add", key=f"addserv_{item_id}_{i}"):
                                add_and_refresh(
                                    food_name=name,
                                    calories=float(computed_kcal or 0),
                                    protein=p_val,
                                    carbs=c_val,
                                    fat=f_val,
                                    serving=f"{servings} serving(s)",
                                    source=source,
                                )

                # thin horizontal separator
                st.markdown("<hr class='result-sep'/>", unsafe_allow_html=True)
        else:
            st.info("No results found. Try different keywords or enable USDA API.")
    else:
        st.info("Type to search foods. Suggestions show instant local matches.")

# DAILY LOG tab
with daily_tab:
    st.header("Daily Log")
    logs = get_logs_for_date(date.today())
    if logs:
        df = df_from_meal_logs(logs)
        st.dataframe(df[["date","food","serving","calories","protein","carbs","fat"]])
        totals2 = calc_macro_sums(df)
        st.metric("Total calories today", f"{totals2['calories']:.0f} kcal")
        st.progress(min(totals2['calories'] / goal_kcal if goal_kcal else 0, 1.0))
    else:
        st.info("No meals logged today.")

# PROGRESS tab
with progress_tab:
    st.header("Progress")
    end = st.date_input("End date", value=date.today(), key="prog_end")
    start = st.date_input("Start date", value=end - timedelta(days=29), key="prog_start")
    if start > end:
        st.error("Start must be before End")
    else:
        rows = get_logs_between(start, end)
        if rows:
            df = df_from_meal_logs(rows)
            daily = df.groupby("date").sum().reset_index().sort_values("date")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily["date"], y=daily["calories"], mode="lines+markers", line=dict(color="#36a2eb")))
            fig.update_layout(title="Calories over time", paper_bgcolor="#0f1113", plot_bgcolor="#0f1113", font=dict(color='#d6d7db'))
            st.plotly_chart(fig, use_container_width=True)
            if set(["protein","carbs","fat"]).issubset(df.columns):
                macros = df.groupby("date")[["protein","carbs","fat"]].sum().reset_index()
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=macros["date"], y=macros["protein"], mode="lines", name="Protein"))
                fig2.add_trace(go.Scatter(x=macros["date"], y=macros["carbs"], mode="lines", name="Carbs"))
                fig2.add_trace(go.Scatter(x=macros["date"], y=macros["fat"], mode="lines", name="Fat"))
                fig2.update_layout(title="Macros over time", paper_bgcolor="#0f1113", plot_bgcolor="#0f1113", font=dict(color='#d6d7db'))
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No logs in this range.")

# EXPORT tab
# ---------------- EXPORT tab (replace existing export_tab block) ----------------
with export_tab:
    st.header("Export")

    end = st.date_input("Export end date", value=date.today(), key="export_end")
    start = st.date_input("Export start date", value=end - timedelta(days=6), key="export_start")
    if start > end:
        st.error("Start must be before End")
    else:
        rows = get_logs_between(start, end)
        df = df_from_meal_logs(rows)

        if df.empty:
            st.info("No data in this range.")
        else:
            # CSV download (existing)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"food_log_{start}_{end}.csv",
                mime="text/csv",
            )

            # PDF download: use reportlab if available
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.pdfgen import canvas
                from reportlab.lib import colors
                REPORTLAB_OK = True
            except Exception:
                REPORTLAB_OK = False

            def generate_pdf_bytes(df_table: pd.DataFrame, start_date, end_date) -> bytes:
                """
                Create a simple PDF with header and a table of the rows.
                Returns bytes of the PDF file.
                """
                from reportlab.lib.pagesizes import letter
                from reportlab.pdfgen import canvas
                from reportlab.lib import colors
                from reportlab.platypus import Table, TableStyle
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                from reportlab.lib.styles import getSampleStyleSheet
                import io

                buf = io.BytesIO()
                doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=30,leftMargin=30, topMargin=30,bottomMargin=18)
                elements = []
                styles = getSampleStyleSheet()
                title = Paragraph(f"BiteWise — Food Log ({start_date} to {end_date})", styles["Title"])
                elements.append(title)
                elements.append(Spacer(1, 12))

                # prepare table data (header + rows)
                header = ["Date", "Food", "Serving", "Calories", "Protein", "Carbs", "Fat"]
                table_data = [header]
                # ensure we have strings for the table
                for _, row in df_table.iterrows():
                    table_data.append([
                        str(row.get("date", "")),
                        str(row.get("food", "")),
                        str(row.get("serving", "")),
                        str(int(row.get("calories", 0))) if not pd.isna(row.get("calories", None)) else "",
                        f"{row.get('protein', '')}",
                        f"{row.get('carbs', '')}",
                        f"{row.get('fat', '')}",
                    ])

                # build table
                tbl = Table(table_data, repeatRows=1, hAlign="LEFT")
                tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#333333")),
                    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                    ("FONTSIZE", (0,0), (-1,0), 10),
                    ("BOTTOMPADDING", (0,0), (-1,0), 6),
                    ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
                    ("FONTSIZE", (0,1), (-1,-1), 9),
                ]))

                elements.append(tbl)
                doc.build(elements)
                buf.seek(0)
                return buf.getvalue()

            if REPORTLAB_OK:
                try:
                    pdf_bytes = generate_pdf_bytes(df, start, end)
                    st.download_button(
                        label="Download PDF",
                        data=pdf_bytes,
                        file_name=f"food_log_{start}_{end}.pdf",
                        mime="application/pdf",
                    )
                except Exception as e:
                    st.error(f"Failed to generate PDF: {e}")
                    with st.expander("Show PDF error"):
                        import traceback
                        st.code(traceback.format_exc())
            else:
                st.info("Install `reportlab` to enable PDF export. Add `reportlab` to requirements.txt and reinstall.")

