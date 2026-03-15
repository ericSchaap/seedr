"""
Seedr — Tournament Schedule Optimizer
======================================
Streamlit web app with athlete-friendly UX.

Usage:
    cd tennis-tournament-optimizer
    streamlit run src/app/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os
import time

# Add modeling directory to path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.join(APP_DIR, '..', 'modeling')
PROJECT_ROOT = os.path.join(APP_DIR, '..', '..')
sys.path.insert(0, MODELING_DIR)

from seasonal_optimizer import SeasonalOptimizer
from travel_costs import TravelCostModel, COUNTRY_CONTINENT
from points_to_rank import PointsRankMapper

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="Seedr — Tournament Optimizer",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# DESIGN TOKENS  (matching the React prototype)
# ==============================================================================
COLORS = {
    'bg': '#FAFAF7', 'card': '#FFFFFF', 'text': '#1A1A18',
    'muted': '#7A7A72', 'light': '#A8A8A0', 'border': '#E8E8E2',
    'accent': '#2D6A4F', 'accent_light': '#D8F3DC', 'accent_mid': '#52B788',
    'warm': '#F7F5EE', 'warm_border': '#E8E4D8',
    'amber': '#D4A017', 'red': '#C1121F',
}

SURFACE_COLORS = {
    'Clay': '#C17817', 'Hard': '#3B7CB8', 'Grass': '#2D8A4E',
    'Hard Indoor': '#7B68EE', 'Carpet': '#DAA520',
}

SURFACE_EMOJI = {
    'Clay': '🟤', 'Hard': '🔵', 'Hard Indoor': '🟣',
    'Grass': '🟢', 'Carpet': '🟠',
}

COUNTRY_NAMES = {
    'ARG': 'Argentina', 'AUS': 'Australia', 'AUT': 'Austria', 'BEL': 'Belgium',
    'BRA': 'Brazil', 'BUL': 'Bulgaria', 'CAN': 'Canada', 'CHI': 'Chile',
    'CHN': 'China', 'COL': 'Colombia', 'CRO': 'Croatia', 'CZE': 'Czech Republic',
    'DEN': 'Denmark', 'ECU': 'Ecuador', 'EGY': 'Egypt', 'ESP': 'Spain',
    'FIN': 'Finland', 'FRA': 'France', 'GBR': 'Great Britain', 'GEO': 'Georgia',
    'GER': 'Germany', 'GRE': 'Greece', 'HUN': 'Hungary', 'IND': 'India',
    'INA': 'Indonesia', 'ITA': 'Italy', 'JPN': 'Japan', 'KAZ': 'Kazakhstan',
    'KOR': 'South Korea', 'MAR': 'Morocco', 'MEX': 'Mexico', 'NED': 'Netherlands',
    'NOR': 'Norway', 'NZL': 'New Zealand', 'PER': 'Peru', 'POL': 'Poland',
    'POR': 'Portugal', 'ROU': 'Romania', 'RSA': 'South Africa', 'RUS': 'Russia',
    'SRB': 'Serbia', 'SLO': 'Slovenia', 'SVK': 'Slovakia', 'SUI': 'Switzerland',
    'SWE': 'Sweden', 'TUN': 'Tunisia', 'TUR': 'Turkey', 'UKR': 'Ukraine',
    'URU': 'Uruguay', 'USA': 'United States', 'UZB': 'Uzbekistan',
}
for code in COUNTRY_CONTINENT:
    if code not in COUNTRY_NAMES:
        COUNTRY_NAMES[code] = code

SURFACE_SEASONS = {
    'Australian Hard (Jan–Mar)': (1, 12),
    'Clay Season (Apr–Jun)': (14, 24),
    'Grass Season (Jun–Jul)': (24, 28),
    'US Hard (Jul–Sep)': (28, 39),
    'Fall Indoor (Oct–Nov)': (40, 47),
    'Custom range': None,
}

TEMPLATE_YEAR = 2025
N_SCHEDULES = 200
N_SIMS_TOURNAMENT = 400
N_SIMS_SCHEDULE = 2000


# ==============================================================================
# CUSTOM CSS
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&family=Fraunces:opsz,wght@9..144,300;9..144,500;9..144,700&display=swap');

    /* Global overrides */
    .stApp { background-color: #FAFAF7; }
    h1, h2, h3, h4 { font-family: 'Fraunces', serif !important; color: #1A1A18; }
    p, span, div, li { font-family: 'DM Sans', sans-serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif; font-size: 13px;
        padding: 8px 16px; border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background-color: white; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }

    /* Seedr card */
    .seedr-card {
        background: white; border: 1px solid #E8E8E2; border-radius: 16px;
        padding: 20px 24px; margin-bottom: 16px;
    }
    .seedr-card-active {
        background: white; border: 1.5px solid #2D6A4F; border-radius: 16px;
        padding: 20px 24px; margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(45,106,79,0.12);
    }

    /* Badge */
    .seedr-badge {
        display: inline-block; font-family: 'DM Sans', sans-serif;
        font-size: 11px; font-weight: 600; padding: 3px 10px;
        border-radius: 20px;
    }

    /* Disclaimer */
    .seedr-disclaimer {
        background: #F7F5EE; border: 1px solid #E8E4D8; border-radius: 12px;
        padding: 12px 16px; font-size: 13px; color: #7A7A72; line-height: 1.5;
    }

    /* Metric tile */
    .seedr-metric {
        background: #F7F7F2; border-radius: 10px; padding: 10px 8px;
        text-align: center;
    }
    .seedr-metric-label { font-size: 10px; color: #A8A8A0; margin-bottom: 3px; }
    .seedr-metric-value { font-size: 16px; font-weight: 700; color: #1A1A18; }

    /* Cost bar segment */
    .cost-segment { border-radius: 4px; height: 8px; display: inline-block; }

    /* Section label */
    .section-label {
        font-size: 11px; font-weight: 500; color: #7A7A72;
        text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 8px;
    }

    /* Estimates note */
    .estimates-note {
        font-size: 11px; color: #A8A8A0; font-style: italic; line-height: 1.4;
        margin-top: 6px;
    }

    /* Hide default Streamlit metric chrome for cleaner look */
    [data-testid="stMetricDelta"] svg { display: none; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# HELPERS
# ==============================================================================

def format_currency(val, symbol='€'):
    if val >= 0:
        return f"{symbol}{val:,.0f}"
    return f"-{symbol}{abs(val):,.0f}"


def get_data_path():
    return os.path.join(PROJECT_ROOT, 'data', 'processed',
                        'atp_clean_both_ranked.csv')


@st.cache_resource(show_spinner="Loading optimizer engine...")
def load_optimizer(player_country):
    data_path = get_data_path()
    if not os.path.exists(data_path):
        return None
    optimizer = SeasonalOptimizer(player_country=player_country)
    optimizer.load_calendar(data_path, year=TEMPLATE_YEAR)
    return optimizer


def generate_schedule_name(sched, idx, tournament_evs, player_rank):
    """Generate a descriptive 2-3 word name for a schedule."""
    tournaments = sched.get('schedule', [])
    if not tournaments:
        return f"Schedule {idx + 1}"

    categories = [t.get('category', '') for _, t in tournaments]
    has_atp = any('ATP' in c or 'Grand Slam' in c for c in categories)
    has_challenger = any('Challenger' in c for c in categories)
    has_itf = any(c.startswith('M15') or c.startswith('M25') for c in categories)

    challenger_pct = sum(1 for c in categories if 'Challenger' in c) / len(categories)
    atp_pct = sum(1 for c in categories if 'ATP' in c or 'Grand Slam' in c) / len(categories)

    if atp_pct >= 0.4:
        tier_word = "ATP Push"
    elif challenger_pct >= 0.6:
        tier_word = "Challenger Circuit"
    elif has_itf and not has_atp:
        tier_word = "ITF Grind"
    elif has_atp and has_challenger:
        tier_word = "Mixed Tier"
    else:
        tier_word = "Balanced"

    surfaces = [t.get('surface', '') for _, t in tournaments]
    clay_pct = sum(1 for s in surfaces if 'Clay' in s) / len(surfaces)
    hard_pct = sum(1 for s in surfaces if 'Hard' in s) / len(surfaces)

    if clay_pct >= 0.8:
        surf_word = "Clay"
    elif hard_pct >= 0.8:
        surf_word = "Hard Court"
    elif clay_pct >= 0.5:
        surf_word = "Clay-Heavy"
    else:
        surf_word = "Mixed Surface"

    n_tournaments = len(tournaments)
    weeks = [w for w, _ in tournaments]
    week_span = max(weeks) - min(weeks) + 1 if weeks else 1
    density = n_tournaments / week_span

    exp_rank = sched.get('expected_final_rank', player_rank)
    rank_improvement = player_rank - exp_rank

    if density >= 0.7:
        style_word = "Packed"
    elif density <= 0.4:
        style_word = "Relaxed"
    elif rank_improvement > 30:
        style_word = "Aggressive"
    elif sched.get('net_prize', 0) > 0:
        style_word = "Profitable"
    else:
        style_word = "Steady"

    if surf_word in ('Clay', 'Hard Court', 'Mixed Surface'):
        return f"{style_word} {surf_word}"
    else:
        return f"{style_word} {tier_word}"


def generate_schedule_badge(sched, idx, all_schedules):
    """Assign a badge based on how this schedule compares to others."""
    if idx == 0:
        return "🟢 Best Overall", COLORS['accent'], COLORS['accent_light']

    # Check if it has highest p90 points
    p90s = [s['points_p90'] for s in all_schedules]
    if sched['points_p90'] >= max(p90s):
        return "🔵 Highest Ceiling", '#2E6AAB', '#DBEAFE'

    # Check if it has lowest cost
    costs = [s['total_cost'] for s in all_schedules]
    if sched['total_cost'] <= min(costs):
        return "🟡 Lowest Cost", COLORS['amber'], '#FFF3CD'

    # Check most consistent (smallest IQR)
    iqrs = [s['points_p90'] - s['points_p10'] for s in all_schedules]
    sched_iqr = sched['points_p90'] - sched['points_p10']
    if sched_iqr <= min(iqrs):
        return "🟣 Most Consistent", '#7B68EE', '#EDE9FE'

    return f"📋 Option {idx + 1}", COLORS['muted'], '#F0F0EA'


def compute_per_tournament_costs(schedule, player_country, home_city=None):
    """Compute flight/hotel/entry breakdown using distance-based EU Commission model."""
    travel_model = TravelCostModel(player_country, home_city=home_city)
    info = travel_model.get_schedule_travel_info(schedule)
    per_tournament = []
    for detail in info['per_tournament']:
        per_tournament.append({
            'entry': detail.get('entry', 0),
            'travel': detail.get('flight', 0),
            'hotel': detail.get('hotel', 0),
            'total': detail.get('cost', 0),
        })
    return per_tournament


# ==============================================================================
# PLOTLY CHART BUILDERS
# ==============================================================================

def build_range_bar(p10, p25, p50, p75, p90, expected=None,
                    color_from='#B7E4C7', color_to='#52B788',
                    show_zero=False, height=50, width=None):
    """Build a Plotly figure for a single range bar (matching prototype)."""
    fig = go.Figure()
    bar_y = 0.5
    bar_h = 0.35

    fig.add_shape(type='rect', x0=p10, x1=p90, y0=bar_y - 0.08, y1=bar_y + 0.08,
                  fillcolor='#DDDDD6', line_width=0, layer='below')
    fig.add_shape(type='rect', x0=p25, x1=p75, y0=bar_y - bar_h, y1=bar_y + bar_h,
                  fillcolor=color_from, opacity=0.6, line_width=0)
    # Median tick
    fig.add_shape(type='line', x0=p50, x1=p50,
                  y0=bar_y - bar_h * 0.7, y1=bar_y + bar_h * 0.7,
                  line=dict(color='rgba(45,106,79,0.4)', width=2))

    if show_zero and p10 < 0 < p90:
        fig.add_shape(type='line', x0=0, x1=0, y0=0, y1=1,
                      line=dict(color='#7A7A72', width=1.5, dash='dot'))
        fig.add_annotation(x=0, y=1.05, text='break even', showarrow=False,
                           font=dict(size=9, color='#7A7A72', family='DM Sans'))

    if expected is not None:
        fig.add_trace(go.Scatter(
            x=[expected], y=[bar_y], mode='markers',
            marker=dict(size=14, color=COLORS['accent'],
                        line=dict(color='white', width=2.5)),
            hovertemplate=f'Expected: {expected}<extra></extra>',
            showlegend=False,
        ))

    fig.update_layout(
        height=height, width=width,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, range=[p10 - (p90 - p10) * 0.05,
                                          p90 + (p90 - p10) * 0.05]),
        yaxis=dict(visible=False, range=[0, 1.1]),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def build_comparison_chart(schedules, schedule_names, metric='points'):
    """Build aligned range bars for all schedules on a shared scale."""
    if metric == 'points':
        data = [{
            'name': n,
            'p10': s['points_p10'], 'p25': s['points_p25'],
            'p50': s['points_p50'],
            'p75': s['points_p75'],
            'p90': s['points_p90'],
            'expected': s['expected_points'],
        } for s, n in zip(schedules, schedule_names)]
        color_band = '#52B788'
        show_zero = False
    else:  # financial — compute net from actual prize percentiles
        data = [{
            'name': n,
            'p10': s['prize_p10'] - s['total_cost'],
            'p25': s['prize_p25'] - s['total_cost'],
            'p50': s['expected_prize'] - s['total_cost'],
            'p75': s['prize_p75'] - s['total_cost'],
            'p90': s['prize_p90'] - s['total_cost'],
            'expected': s['net_prize'],
        } for s, n in zip(schedules, schedule_names)]
        color_band = '#DAA520'
        show_zero = True

    global_min = min(d['p10'] for d in data)
    global_max = max(d['p90'] for d in data)
    pad = (global_max - global_min) * 0.06

    fig = go.Figure()
    n = len(data)

    for i, d in enumerate(data):
        y = n - i
        # Whisker: p10 → p90
        fig.add_shape(type='line', x0=d['p10'], x1=d['p90'],
                      y0=y, y1=y,
                      line=dict(color='#DDDDD6', width=4))
        # Band: p25 → p75
        fig.add_shape(type='rect',
                      x0=d['p25'], x1=d['p75'],
                      y0=y - 0.28, y1=y + 0.28,
                      fillcolor=color_band, opacity=0.5, line_width=0)
        # Median tick
        fig.add_shape(type='line', x0=d['p50'], x1=d['p50'],
                      y0=y - 0.22, y1=y + 0.22,
                      line=dict(color='rgba(45,106,79,0.4)', width=2))
        # Expected dot
        fig.add_trace(go.Scatter(
            x=[d['expected']], y=[y], mode='markers',
            marker=dict(size=13, color=COLORS['accent'],
                        line=dict(color='white', width=2.5)),
            showlegend=False,
            hovertemplate=(
                f"<b>{d['name']}</b><br>"
                f"Expected: {d['expected']:.1f}<br>"
                f"Range: {d['p10']:.0f} – {d['p90']:.0f}"
                "<extra></extra>"
            ),
        ))

    if show_zero and global_min < 0 < global_max:
        fig.add_shape(type='line', x0=0, x1=0,
                      y0=0.3, y1=n + 0.7,
                      line=dict(color='#7A7A72', width=1.5, dash='dot'))
        fig.add_annotation(x=0, y=n + 0.85, text='break even', showarrow=False,
                           font=dict(size=10, color='#7A7A72', family='DM Sans'))

    fig.update_layout(
        height=60 + n * 52,
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(
            range=[global_min - pad, global_max + pad],
            showgrid=True, gridcolor='rgba(128,128,128,0.1)',
            zeroline=False,
            tickfont=dict(size=10, color='#A8A8A0', family='DM Sans'),
        ),
        yaxis=dict(
            tickvals=list(range(1, n + 1)),
            ticktext=list(reversed([d['name'] for d in data])),
            tickfont=dict(size=12, color='#1A1A18', family='DM Sans'),
        ),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans'),
    )
    return fig


def build_round_journey(round_probs, height=120):
    """Build the 'how far you'll likely go' bar chart."""
    # Simulation uses '1/16' notation; normalize to a single format for lookup
    SIM_TO_DISPLAY = {
        '1/64': 'R128', '1/32': 'R64', '1/16': 'R32', '1/8': 'R16',
        'QF': 'QF', 'SF': 'SF', 'F': 'F', 'W': 'W',
        # Also accept display-format keys directly
        'R128': 'R128', 'R64': 'R64', 'R32': 'R32', 'R16': 'R16',
    }
    DISPLAY_LABELS = {
        'R128': 'R128', 'R64': 'R64', 'R32': 'R32', 'R16': 'R16',
        'QF': 'QF', 'SF': 'SF', 'F': 'Final', 'W': 'Win',
    }
    display_order = ['R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F', 'W']

    # Remap round_probs to display keys
    mapped = {}
    for k, v in round_probs.items():
        dk = SIM_TO_DISPLAY.get(k)
        if dk:
            mapped[dk] = v

    rounds, probs, labels = [], [], []
    for r in display_order:
        if r in mapped:
            rounds.append(r)
            probs.append(mapped[r])
            labels.append(DISPLAY_LABELS[r])

    colors = [f'rgba(45,106,79,{0.25 + p * 0.75})' for p in probs]
    text = [f'{p:.0%}' for p in probs]

    fig = go.Figure(go.Bar(
        x=labels, y=probs, marker_color=colors,
        text=text, textposition='outside',
        textfont=dict(size=11, color=COLORS['accent'], family='DM Sans'),
        hovertemplate='%{x}: %{y:.1%}<extra></extra>',
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=5, b=25),
        xaxis=dict(tickfont=dict(size=10, color='#A8A8A0', family='DM Sans')),
        yaxis=dict(visible=False, range=[0, max(probs) * 1.25] if probs else [0, 1]),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        bargap=0.35,
    )
    return fig


def build_calendar_strip(schedule, start_week, end_week):
    """Build a compact Plotly calendar showing tournament weeks."""
    playing = {w: t for w, t in schedule}

    fig = go.Figure()
    weeks = list(range(start_week, end_week + 1))
    n = len(weeks)

    for i, w in enumerate(weeks):
        if w in playing:
            t = playing[w]
            surf = t.get('surface', '')
            color = SURFACE_COLORS.get(surf, '#888888')
            name = t.get('tournament_name', '?')
            cat = t.get('category', '?')
            country = COUNTRY_NAMES.get(t.get('country', ''), '?')
            hover = f"<b>Week {w}: {name}</b><br>{cat} · {surf}<br>{country}"
        else:
            color = '#F0F0EA'
            hover = f"Week {w}: rest"

        fig.add_trace(go.Scatter(
            x=[i], y=[0], mode='markers+text',
            marker=dict(size=28, color=color, symbol='square',
                        line=dict(color='#E8E8E2' if w not in playing else color, width=1)),
            text=str(w),
            textfont=dict(size=10, color='white' if w in playing else '#A8A8A0',
                          family='DM Sans'),
            textposition='middle center',
            hovertemplate=hover + '<extra></extra>',
            showlegend=False,
        ))

    fig.update_layout(
        height=55, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, range=[-0.8, n - 0.2]),
        yaxis=dict(visible=False, range=[-0.8, 0.8]),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def build_cost_breakdown_bar(entry, travel, hotel, total):
    """Build a proportional stacked cost bar."""
    fig = go.Figure()
    segments = [
        ('Entry', entry, '#8B7355'),
        ('Travel', travel, '#A0937D'),
        ('Hotel', hotel, '#C4B8A5'),
    ]
    x_pos = 0
    for label, val, color in segments:
        if val > 0 and total > 0:
            width = val / total
            fig.add_shape(type='rect',
                          x0=x_pos, x1=x_pos + width,
                          y0=0.2, y1=0.8,
                          fillcolor=color, line_width=0)
            if width > 0.12:
                fig.add_annotation(
                    x=x_pos + width / 2, y=0.5,
                    text=f'{label}<br>€{val}',
                    showarrow=False,
                    font=dict(size=9, color='white', family='DM Sans'),
                    align='center',
                )
            x_pos += width

    fig.update_layout(
        height=45, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


# ==============================================================================
# DISPLAY FUNCTIONS
# ==============================================================================

def render_disclaimer():
    st.markdown("""
    <div class="seedr-disclaimer">
        🎲 &nbsp; These schedules are based on simulating thousands of tournament
        draws using historical data from players at your level. They show what's
        <em>likely</em>, not what's certain. On range bars: the
        <span style="display:inline-block;width:8px;height:8px;border-radius:50%;
        background:#2D6A4F;border:1.5px solid white;vertical-align:middle;
        box-shadow:0 0 0 1px #2D6A4F;margin:0 2px"></span>
        dot is the expected value, the colored band covers the middle 50% of
        outcomes, and the full bar spans the realistic range.
    </div>
    """, unsafe_allow_html=True)


def render_metrics_row(sched, player_rank):
    rank_delta = player_rank - sched['expected_final_rank']
    delta_str = f' ↑{rank_delta:.0f}' if rank_delta > 0 else ''
    cols = st.columns(4)
    metrics = [
        ("Tournaments", str(sched.get('n_tournaments', len(sched.get('schedule', []))))),
        ("Expected points", f"~{sched['expected_points']:.1f}"),
        ("Projected rank", f"#{sched['expected_final_rank']:.0f}{delta_str}"),
        ("Est. total cost", format_currency(sched['total_cost'])),
    ]
    for col, (label, value) in zip(cols, metrics):
        col.markdown(f"""
        <div class="seedr-metric">
            <div class="seedr-metric-label">{label}</div>
            <div class="seedr-metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)


def render_schedule_overview(sched, sched_name, badge_text, badge_color,
                             badge_bg, player_rank, sw, ew, tournament_details):
    """Render a full schedule card with range bars and calendar."""
    # Badge + name
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
        <span style="font-family:'Fraunces',serif;font-size:18px;font-weight:600;color:#1A1A18">
            {sched_name}
        </span>
        <span class="seedr-badge" style="color:{badge_color};background:{badge_bg}">
            {badge_text}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Tagline
    tagline = _generate_tagline(sched, player_rank)
    st.markdown(f'<p style="color:#7A7A72;font-style:italic;font-size:13px;margin:0 0 12px">'
                f'{tagline}</p>', unsafe_allow_html=True)

    # Calendar strip
    st.plotly_chart(build_calendar_strip(sched['schedule'], sw, ew),
                    use_container_width=True, key=f"cal_{sched_name}")

    # Metrics row
    render_metrics_row(sched, player_rank)

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    # Ranking points range
    st.markdown('<div class="section-label">Ranking points range</div>',
                unsafe_allow_html=True)
    pts_exp = sched['expected_points']
    pts_p10 = sched['points_p10']
    pts_p25 = sched['points_p25']
    pts_p50 = sched['points_p50']
    pts_p75 = sched['points_p75']
    pts_p90 = sched['points_p90']

    lcol, rcol = st.columns([5, 1])
    with lcol:
        st.markdown(f'<div style="font-size:11px;font-weight:700;color:#2D6A4F;'
                    f'margin-bottom:2px">expected: ~{pts_exp:.1f} pts</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            build_range_bar(pts_p10, pts_p25, pts_p50, pts_p75, pts_p90,
                            expected=pts_exp, height=45),
            use_container_width=True, key=f"pts_{sched_name}")
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;margin-top:-8px">
            <span style="font-size:10px;color:#7A7A72">{pts_p10:.0f} pts</span>
            <span style="font-size:10px;color:#7A7A72">{pts_p90:.0f} pts</span>
        </div>
        """, unsafe_allow_html=True)

    # Financial outcome range
    st.markdown('<div class="section-label" style="margin-top:12px">'
                'Financial outcome range</div>', unsafe_allow_html=True)
    net = sched['net_prize']
    # Use actual percentiles from optimizer (now includes prize_p25/p75)
    fin_p10 = sched['prize_p10'] - sched['total_cost']
    fin_p25 = sched['prize_p25'] - sched['total_cost']
    fin_p50 = sched['expected_prize'] - sched['total_cost']
    fin_p75 = sched['prize_p75'] - sched['total_cost']
    fin_p90 = sched['prize_p90'] - sched['total_cost']

    net_label = f"+€{net:.0f}" if net >= 0 else f"-€{abs(net):.0f}"
    st.markdown(f'<div style="font-size:11px;font-weight:700;color:#2D6A4F;'
                f'margin-bottom:2px">expected: {net_label}</div>',
                unsafe_allow_html=True)
    st.plotly_chart(
        build_range_bar(fin_p10, fin_p25, fin_p50, fin_p75, fin_p90,
                        expected=net, color_from='#FFE8A0', color_to='#B7E4C7',
                        show_zero=True, height=55),
        use_container_width=True, key=f"fin_{sched_name}")
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;margin-top:-8px">
        <span style="font-size:10px;color:#7A7A72">€{abs(fin_p10):.0f} loss</span>
        <span style="font-size:10px;color:#7A7A72">€{fin_p90:.0f} gain</span>
    </div>
    """, unsafe_allow_html=True)


def render_tournament_detail(tournament, cost_breakdown, tournament_details):
    """Render a single tournament tab with journey, points range, cost breakdown."""
    name = tournament.get('tournament_name', '?')
    cat = tournament.get('category', '?')
    surface = tournament.get('surface', '?')
    country = tournament.get('country', '?')
    country_name = COUNTRY_NAMES.get(country, country) if isinstance(country, str) else '?'
    surf_color = SURFACE_COLORS.get(surface, '#888')

    details = tournament_details.get(name, {})
    exp_pts = details.get('expected_points', 0)
    exp_prize = details.get('expected_prize', 0)
    round_probs = details.get('round_probs', {})
    pts_ci = details.get('points_ci', {})

    # Header
    hcol, vcol = st.columns([3, 1])
    with hcol:
        st.markdown(f"""
        <h4 style="margin:0 0 4px;font-size:16px">{name}</h4>
        <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#7A7A72">
            <span style="width:8px;height:8px;border-radius:50%;background:{surf_color};
            display:inline-block"></span>
            {cat} · {surface} · {country_name}
        </div>
        """, unsafe_allow_html=True)
    with vcol:
        st.markdown(f"""
        <div style="text-align:right">
            <div style="font-size:18px;font-weight:700;color:#2D6A4F">~{exp_pts:.1f} pts</div>
            <div style="font-size:11px;color:#A8A8A0">expected</div>
        </div>
        """, unsafe_allow_html=True)

    # How far you'll likely go
    if round_probs:
        st.markdown('<div class="section-label" style="margin-top:12px">'
                    'How far you\'ll likely go</div>', unsafe_allow_html=True)
        st.plotly_chart(build_round_journey(round_probs),
                        use_container_width=True, key=f"rj_{name}")

    # Points range
    if pts_ci:
        st.markdown('<div class="section-label">Ranking points range</div>',
                    unsafe_allow_html=True)
        p10 = pts_ci.get('p10', 0)
        p25 = pts_ci.get('p25', 0)
        p50 = pts_ci.get('p50', 0)
        p75 = pts_ci.get('p75', 0)
        p90 = pts_ci.get('p90', 0)
        st.markdown(f'<div style="font-size:11px;font-weight:700;color:#2D6A4F;'
                    f'margin-bottom:2px">expected: ~{exp_pts:.1f} pts</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            build_range_bar(p10, p25, p50, p75, p90, expected=exp_pts, height=40),
            use_container_width=True, key=f"tpts_{name}")
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;margin-top:-8px">
            <span style="font-size:10px;color:#7A7A72">{p10:.0f} pts</span>
            <span style="font-size:10px;color:#7A7A72">{p90:.0f} pts</span>
        </div>
        """, unsafe_allow_html=True)

    # Cost breakdown
    entry = cost_breakdown['entry']
    travel = cost_breakdown['travel']
    hotel = cost_breakdown['hotel']
    total = cost_breakdown['total']

    st.markdown('<div class="section-label" style="margin-top:12px">'
                'Trip cost breakdown</div>', unsafe_allow_html=True)
    st.plotly_chart(build_cost_breakdown_bar(entry, travel, hotel, total),
                    use_container_width=True, key=f"cost_{name}")

    c1, c2, c3 = st.columns(3)
    for col, label, val, clr in [
        (c1, 'Entry', entry, '#8B7355'),
        (c2, 'Travel', travel, '#A0937D'),
        (c3, 'Hotel', hotel, '#C4B8A5'),
    ]:
        col.markdown(f"""
        <div style="text-align:center">
            <div style="font-size:10px;color:#A8A8A0">{label}</div>
            <div style="font-size:13px;font-weight:600;color:#7A7A72">
                {'€' + str(val) if val > 0 else '—'}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
    padding:8px 12px;background:#F5F5F0;border-radius:8px;margin-top:8px">
        <span style="font-size:12px;color:#7A7A72;font-weight:500">Total trip cost</span>
        <span style="font-size:15px;font-weight:700;color:#1A1A18">~€{total}</span>
    </div>
    <div style="display:flex;justify-content:space-between;align-items:center;
    padding:8px 12px;margin-top:4px">
        <span style="font-size:11px;color:#A8A8A0">Expected prize money</span>
        <span style="font-size:13px;font-weight:600;color:#2D6A4F">~€{exp_prize:.0f}</span>
    </div>
    <div class="estimates-note">
        Travel and hotel are rough estimates based on distance and typical rates —
        your actual costs may differ.
    </div>
    """, unsafe_allow_html=True)


def render_season_summary(sched, sched_name):
    """Dark green season summary card."""
    fin_net = sched['net_prize']
    net_str = f"+€{fin_net:.0f}" if fin_net >= 0 else f"-€{abs(fin_net):.0f}"
    fin_p10 = sched.get('prize_p10', 0) - sched['total_cost']
    fin_p90 = sched.get('prize_p90', 0) - sched['total_cost']

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#2D6A4F 0%,#1B4332 100%);
    border-radius:16px;padding:22px 24px;color:white;margin-top:20px">
        <h3 style="margin:0 0 4px;font-size:15px;color:white">{sched_name} — Season snapshot</h3>
        <p style="margin:0 0 14px;font-size:11px;opacity:0.65;color:white">
            Expected outcomes for this schedule</p>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px">
            <div style="text-align:center">
                <div style="font-size:9px;opacity:0.55;text-transform:uppercase;
                letter-spacing:0.05em;margin-bottom:3px">Tournaments</div>
                <div style="font-size:17px;font-weight:700">
                    {sched.get('n_tournaments', len(sched.get('schedule', [])))}</div>
            </div>
            <div style="text-align:center">
                <div style="font-size:9px;opacity:0.55;text-transform:uppercase;
                letter-spacing:0.05em;margin-bottom:3px">Expected pts</div>
                <div style="font-size:17px;font-weight:700">~{sched['expected_points']:.1f}</div>
            </div>
            <div style="text-align:center">
                <div style="font-size:9px;opacity:0.55;text-transform:uppercase;
                letter-spacing:0.05em;margin-bottom:3px">Projected rank</div>
                <div style="font-size:17px;font-weight:700">
                    #{sched['expected_final_rank']:.0f}</div>
            </div>
            <div style="text-align:center">
                <div style="font-size:9px;opacity:0.55;text-transform:uppercase;
                letter-spacing:0.05em;margin-bottom:3px">Est. cost</div>
                <div style="font-size:17px;font-weight:700">{format_currency(sched['total_cost'])}</div>
            </div>
        </div>
        <div style="margin-top:14px;padding:10px 14px;background:rgba(255,255,255,0.1);
        border-radius:10px;font-size:11px;opacity:0.7;line-height:1.5">
            Financial range: €{abs(fin_p10):.0f} loss in a tough stretch →
            €{fin_p90:.0f} gain if results go well. Expected net: {net_str}.
        </div>
    </div>
    """, unsafe_allow_html=True)


def _generate_tagline(sched, player_rank):
    """Generate a 1-line coaching tagline for a schedule."""
    n = sched.get('n_tournaments', len(sched.get('schedule', [])))
    rank_delta = player_rank - sched['expected_final_rank']
    net = sched['net_prize']

    if rank_delta > 20 and net > 0:
        return "Strong ranking push that should pay for itself"
    elif rank_delta > 20:
        return "Ambitious ranking climb — results need to go your way financially"
    elif net > 0 and n <= 4:
        return "Fewer tournaments, all close to home — lower spend, easier on the body"
    elif net > 0:
        return "Consistent points with manageable travel — your most balanced option"
    elif n >= 6:
        return "Packed schedule with many bites at the apple"
    else:
        return "Tougher fields, bigger upside — high reward if results go your way"


# ==============================================================================
# SIDEBAR — INPUTS (kept from original)
# ==============================================================================

with st.sidebar:
    st.title("🎾 Seedr")
    st.caption("Your season, optimized")
    st.divider()

    st.markdown("**Tour:** ATP")
    st.caption("_WTA support coming soon_")
    st.divider()

    st.subheader("Player Profile")

    player_rank = st.number_input(
        "Current ranking", min_value=1, max_value=2500,
        value=250, step=10,
        help="Your current ATP singles ranking")

    mapper = PointsRankMapper()
    estimated_points = mapper.rank_to_points(player_rank)

    player_points = st.number_input(
        "Current points", min_value=0, max_value=10000,
        value=max(0, int(estimated_points)),
        help="Auto-estimated from rank — adjust if you know your exact total")

    sorted_countries = sorted(COUNTRY_NAMES.items(), key=lambda x: x[1])
    country_labels = [f"{name} ({code})" for code, name in sorted_countries]
    country_codes = [code for code, _ in sorted_countries]
    default_idx = country_codes.index('FRA') if 'FRA' in country_codes else 0

    country_idx = st.selectbox(
        "Home country",
        range(len(country_labels)),
        format_func=lambda i: country_labels[i],
        index=default_idx,
        help="Used for travel cost and geographic scheduling")
    player_country = country_codes[country_idx]

    home_city = st.text_input(
        "Home city",
        value="",
        placeholder="e.g. Paris, Vienna, Buenos Aires",
        help="For accurate flight cost estimates (distances to tournament cities)")

    st.divider()

    st.subheader("Weekly Overhead")
    st.caption("_Costs per week on tour (set to 0 if not applicable)_")
    oh_col1, oh_col2 = st.columns(2)
    with oh_col1:
        overhead_coach = st.number_input(
            "Coach (€/week)", min_value=0, max_value=5000,
            value=0, step=100)
        overhead_physio = st.number_input(
            "Physio (€/week)", min_value=0, max_value=2000,
            value=0, step=50)
    with oh_col2:
        overhead_food = st.number_input(
            "Food (€/week)", min_value=0, max_value=1000,
            value=0, step=50)
        overhead_other = st.number_input(
            "Other (€/week)", min_value=0, max_value=2000,
            value=0, step=50)
    weekly_overhead = overhead_coach + overhead_physio + overhead_food + overhead_other
    if weekly_overhead > 0:
        st.caption(f"_Total overhead: €{weekly_overhead:,}/week on tour_")

    st.divider()

    st.subheader("Planning Window")

    season_label = st.selectbox(
        "Surface season",
        list(SURFACE_SEASONS.keys()),
        index=1,
        help="Select a surface season or define a custom week range")

    season_range = SURFACE_SEASONS[season_label]
    if season_range is None:
        col1, col2 = st.columns(2)
        with col1:
            start_week = st.number_input("Start week", 1, 52, 14)
        with col2:
            end_week = st.number_input("End week", 1, 52, 24)
    else:
        start_week, end_week = season_range
        st.caption(f"Weeks {start_week}–{end_week}")

    st.caption(f"_Using {TEMPLATE_YEAR} calendar as template_")
    st.divider()

    st.subheader("Preferences")

    surface_pref = st.selectbox(
        "Surface preference",
        ['Follow season', 'Clay only', 'Hard only', 'Grass only', 'No preference'],
        help="'Follow season' adapts to the time of year")
    surface_pref_map = {
        'Follow season': 'follow_season', 'Clay only': 'clay_only',
        'Hard only': 'hard_only', 'Grass only': 'grass_only',
        'No preference': 'no_preference',
    }

    travel_scope = st.selectbox(
        "Travel scope",
        ['Continental', 'National only', 'Global'],
        help="How far are you willing to travel?")
    travel_scope_map = {
        'Continental': 'continental', 'National only': 'national',
        'Global': 'global',
    }

    has_budget = st.toggle("Set a budget limit", value=False)
    if has_budget:
        max_budget = st.number_input(
            "Maximum season budget (€)", min_value=500, max_value=100000,
            value=8000, step=500,
            help="Total spend cap: travel + entry fees + accommodation")
    else:
        max_budget = None

    st.divider()

    run_clicked = st.button(
        "🚀 Optimize Schedule",
        use_container_width=True,
        type="primary")


# ==============================================================================
# MAIN AREA
# ==============================================================================

budget_str = f" · Budget {format_currency(max_budget)}" if max_budget else ""
st.markdown(f"""
<div style="text-align:center;margin-bottom:20px">
    <h1 style="margin:0;font-size:30px;letter-spacing:-0.02em">Seedr</h1>
    <p style="margin:4px 0 0;font-size:13px;color:#7A7A72">Your season, optimized</p>
</div>
""", unsafe_allow_html=True)

# Player context bar
st.markdown(f"""
<div class="seedr-card" style="display:flex;justify-content:space-between;
align-items:center;flex-wrap:wrap;gap:8px;padding:14px 18px">
    <div>
        <div style="font-size:12px;color:#7A7A72;margin-bottom:2px">Showing schedules for</div>
        <div style="font-size:15px;font-weight:600;color:#1A1A18">
            ATP #{player_rank} · {COUNTRY_NAMES.get(player_country, player_country)} · {season_label}{budget_str}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

data_path = get_data_path()
if not os.path.exists(data_path):
    st.error(
        f"Processed data not found at `{data_path}`.\n\n"
        f"Run the pipeline first:\n\n"
        f"```\npython src/modeling/00_unified_pipeline.py\n```")
    st.stop()


# ==============================================================================
# RUN OPTIMIZER
# ==============================================================================

if run_clicked:
    with st.spinner("Loading optimizer and calendar data..."):
        optimizer = load_optimizer(player_country)

    if optimizer is None:
        st.error("Failed to load optimizer.")
        st.stop()

    if optimizer.player_country != player_country:
        st.cache_resource.clear()
        optimizer = load_optimizer(player_country)

    st.info(
        f"Generating and simulating {N_SCHEDULES} candidate schedules. "
        f"This typically takes 2–4 minutes...")

    progress_bar = st.progress(0, text="Initializing...")
    t_start = time.time()
    progress_bar.progress(10, text="Computing per-tournament expected values...")

    results = optimizer.optimize(
        player_rank=player_rank,
        player_points=player_points,
        planning_start_week=start_week,
        planning_end_week=end_week,
        n_schedules=N_SCHEDULES,
        n_sims_per_tournament=N_SIMS_TOURNAMENT,
        n_sims_per_schedule=N_SIMS_SCHEDULE,
        max_budget=max_budget,
        surface_preference=surface_pref_map[surface_pref],
        travel_scope=travel_scope_map[travel_scope],
        seed=42,
        verbose=False,
    )

    elapsed = time.time() - t_start
    progress_bar.progress(100, text=f"Done in {elapsed:.0f}s!")
    time.sleep(0.5)
    progress_bar.empty()

    st.session_state['results'] = results
    st.session_state['elapsed'] = elapsed
    st.session_state['start_week'] = start_week
    st.session_state['end_week'] = end_week


# ==============================================================================
# DISPLAY RESULTS
# ==============================================================================

if 'results' in st.session_state:
    results = st.session_state['results']
    elapsed = st.session_state.get('elapsed', 0)
    sw = st.session_state.get('start_week', 14)
    ew = st.session_state.get('end_week', 24)

    if 'error' in results:
        st.error(f"Optimization failed: {results['error']}")
        st.stop()

    meta = results['metadata']
    top_schedules = results['top_schedules']
    tournament_details = results.get('tournament_details', {})
    tournament_evs = results.get('tournament_evs', {})
    tournament_accept = results.get('tournament_accept', {})

    # Generate descriptive names + badges
    schedule_names = []
    for i, sched in enumerate(top_schedules):
        name = generate_schedule_name(sched, i, tournament_evs, meta['player_rank'])
        if name in schedule_names:
            name = f"{name} ({i + 1})"
        schedule_names.append(name)

    schedule_badges = [
        generate_schedule_badge(s, i, top_schedules)
        for i, s in enumerate(top_schedules)
    ]

    # Precompute per-tournament costs for all schedules
    schedule_costs = []
    for sched in top_schedules:
        costs = compute_per_tournament_costs(
            sched['schedule'], meta['player_country'],
            home_city=home_city if home_city else None)
        schedule_costs.append(costs)

    # Inject overhead costs into each schedule's totals
    overhead_per_week = {
        'coach': overhead_coach, 'physio': overhead_physio,
        'food': overhead_food, 'other': overhead_other,
    }
    for sched in top_schedules:
        active_weeks = len(set(w for w, _ in sched['schedule']))
        overhead_total = weekly_overhead * active_weeks
        sched['overhead_total'] = overhead_total
        sched['overhead_per_week'] = weekly_overhead
        sched['active_weeks'] = active_weeks
        # Adjust total cost and net prize to include overhead
        sched['total_cost'] = sched['total_cost'] + overhead_total
        sched['net_prize'] = sched['expected_prize'] - sched['total_cost']

    # --- Disclaimer ---
    render_disclaimer()

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

    # --- Compare at a glance ---
    st.markdown("""
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <h3 style="margin:0;font-size:15px">Compare at a glance</h3>
    </div>
    """, unsafe_allow_html=True)

    comp_tab_pts, comp_tab_fin = st.tabs(["📊 Ranking Points", "💰 Financial"])

    with comp_tab_pts:
        fig_comp = build_comparison_chart(
            top_schedules, schedule_names, metric='points')
        st.plotly_chart(fig_comp, use_container_width=True, key="comp_pts")
        st.caption("All bars share the same scale — wider = more variation in outcomes")

    with comp_tab_fin:
        fig_comp_fin = build_comparison_chart(
            top_schedules, schedule_names, metric='financial')
        st.plotly_chart(fig_comp_fin, use_container_width=True, key="comp_fin")
        st.caption("All bars share the same scale — wider = more variation in outcomes")

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    # --- Schedule tabs ---
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <h3 style="margin:0;font-size:15px">Your top schedules</h3>
        <span style="font-size:11px;color:#A8A8A0">
            {len(top_schedules)} options from {meta['n_schedules_generated']} simulated
        </span>
    </div>
    """, unsafe_allow_html=True)

    sched_tabs = st.tabs(schedule_names)

    for idx, (tab, sched, sched_name, badge, costs) in enumerate(
            zip(sched_tabs, top_schedules, schedule_names,
                schedule_badges, schedule_costs)):
        with tab:
            badge_text, badge_color, badge_bg = badge

            render_schedule_overview(
                sched, sched_name, badge_text, badge_color, badge_bg,
                meta['player_rank'], sw, ew, tournament_details)

            st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

            # --- Tournament detail tabs ---
            tournaments = sched['schedule']
            if tournaments:
                tourn_names = []
                for _, t in tournaments:
                    name = t.get('tournament_name', '?')
                    short = name.replace('ITF ', '').replace('Challenger ', 'CH ')
                    surf = t.get('surface', '')
                    emoji = SURFACE_EMOJI.get(surf, '⚪')
                    tourn_names.append(f"{emoji} {short}")

                tourn_tabs = st.tabs(tourn_names)

                for t_idx, (t_tab, (week, tournament)) in enumerate(
                        zip(tourn_tabs, tournaments)):
                    with t_tab:
                        cost_bd = costs[t_idx] if t_idx < len(costs) else {
                            'entry': 0, 'travel': 0, 'hotel': 0, 'total': 0}
                        render_tournament_detail(
                            tournament, cost_bd, tournament_details)

            # Season summary
            render_season_summary(sched, sched_name)

    st.divider()

    # --- All eligible tournaments ---
    with st.expander("📋 All eligible tournaments ranked by expected value"):
        all_rows = []
        for name, ev in sorted(tournament_evs.items(), key=lambda x: -x[1]):
            details = tournament_details.get(name, {})
            accept = tournament_accept.get(name, 1.0)
            all_rows.append({
                'Tournament': name,
                'Expected pts': f"{ev:.1f}",
                'Acceptance': f"{accept:.0%}",
                'Exp. Prize': format_currency(details.get('expected_prize', 0)),
            })
        st.dataframe(pd.DataFrame(all_rows),
                      use_container_width=True, hide_index=True)

    with st.expander("ℹ️ How are schedules ranked?"):
        st.markdown("""
        Schedules are ranked by **expected ranking points** — the
        probability-weighted average across thousands of Monte Carlo
        simulations. Net financial return (prize minus costs) serves
        as a tiebreaker.

        The top results are filtered for **diversity**: if two schedules
        share more than 70% of the same tournaments, only the better
        one is kept. Each schedule gets a descriptive name based on
        its tier mix, surface focus, and strategic profile.
        """)

else:
    # =========================================================================
    # WELCOME SCREEN
    # =========================================================================
    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("""
        ### How it works

        1. **Set your profile** — Enter your ranking, points, and home
           country in the sidebar
        2. **Choose a planning window** — Pick a surface season or custom
           week range
        3. **Set preferences** — Surface, travel scope, optional budget cap
        4. **Hit Optimize** — Seedr generates hundreds of candidate
           schedules and simulates each one thousands of times
        5. **Compare options** — Review the top schedules with range charts,
           calendar views, and cost breakdowns
        """)

    with col_right:
        st.markdown("""
        ### Quick start profiles

        **Rank ~250 clay specialist**
        - Clay Season, Follow season
        - Travel: Continental

        **Rank ~500 developing player**
        - Any season, No preference
        - Budget: €5,000

        **Rank ~100 Slam qualifier**
        - Clay Season, Follow season
        - Travel: Global
        """)

        st.markdown("---")
        st.markdown("""
        ### Current limitations

        - **ATP only** — WTA model not yet validated
        - **Calendar template** — Uses the 2025 schedule
        - **No injury/fatigue modeling** — rest weeks are rule-based
        """)

    st.info(
        "👈 Configure your profile in the sidebar and click "
        "**Optimize Schedule** to get started.")
