"""
Seedr — Tennis Tournament Optimizer
====================================
Streamlit web app for data-driven tournament selection.

Usage:
    cd tennis-tournament-optimizer
    streamlit run src/app/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import time

# Add modeling directory to path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.join(APP_DIR, '..', 'modeling')
PROJECT_ROOT = os.path.join(APP_DIR, '..', '..')
sys.path.insert(0, MODELING_DIR)

from seasonal_optimizer import SeasonalOptimizer
from travel_costs import COUNTRY_CONTINENT
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
# CUSTOM STYLING
# ==============================================================================
st.markdown("""
<style>
    /* Clean up default streamlit padding */
    .block-container { padding-top: 2rem; }
    
    /* Schedule card styling */
    .schedule-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #0f3460;
        color: #e0e0e0;
    }
    .schedule-card h3 { color: #e0e0e0; margin-top: 0; }
    
    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Surface color coding */
    .surface-clay { color: #e07b3c; font-weight: bold; }
    .surface-hard { color: #4a90d9; font-weight: bold; }
    .surface-grass { color: #5cb85c; font-weight: bold; }
    .surface-indoor { color: #9b59b6; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CONSTANTS
# ==============================================================================

# Country code -> full name (for display)
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

# Add any missing countries from the data
for code in COUNTRY_CONTINENT:
    if code not in COUNTRY_NAMES:
        COUNTRY_NAMES[code] = code

# Surface seasons (week ranges)
SURFACE_SEASONS = {
    'Australian Hard (Jan-Mar)': (1, 12),
    'Clay Season (Apr-Jun)': (14, 24),
    'Grass Season (Jun-Jul)': (24, 28),
    'US Hard (Jul-Sep)': (28, 39),
    'Fall Indoor (Oct-Nov)': (40, 47),
    'Custom range': None,
}

TOUR_OPTIONS = {
    'ATP (Men)': 'atp',
    'WTA (Women)': 'wta',
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def surface_badge(surface):
    """Return a colored surface label."""
    colors = {
        'Clay': '🟤', 'Hard': '🔵', 'Hard Indoor': '🟣',
        'Grass': '🟢', 'Carpet': '🟠',
    }
    return f"{colors.get(surface, '⚪')} {surface}"


def format_currency(val):
    """Format a number as currency."""
    if val >= 0:
        return f"${val:,.0f}"
    return f"-${abs(val):,.0f}"


def get_data_path(tour_code):
    """Get the path to the processed data file."""
    return os.path.join(PROJECT_ROOT, 'data', 'processed',
                        f'{tour_code}_clean_both_ranked.csv')


@st.cache_resource(show_spinner="Loading optimizer engine...")
def load_optimizer(tour_code, player_country, calendar_year):
    """Load and cache the optimizer with calendar data."""
    data_path = get_data_path(tour_code)
    if not os.path.exists(data_path):
        return None
    
    optimizer = SeasonalOptimizer(player_country=player_country)
    optimizer.load_calendar(data_path, year=calendar_year)
    return optimizer


# ==============================================================================
# SIDEBAR — INPUTS
# ==============================================================================

with st.sidebar:
    st.title("🎾 Seedr")
    st.caption("Tournament Optimizer")
    st.divider()
    
    # --- Tour selection ---
    tour_label = st.selectbox("Tour", list(TOUR_OPTIONS.keys()))
    tour_code = TOUR_OPTIONS[tour_label]
    
    # --- Player info ---
    st.subheader("Player Profile")
    
    player_rank = st.number_input(
        "Current ranking", min_value=1, max_value=2500,
        value=250, step=10,
        help="Your current ATP/WTA singles ranking")
    
    # Auto-estimate points from rank
    mapper = PointsRankMapper()
    estimated_points = mapper.rank_to_points(player_rank)
    
    player_points = st.number_input(
        "Current points", min_value=0, max_value=10000,
        value=max(0, int(estimated_points)),
        help="Your current ranking points (auto-estimated from rank)")
    
    # Country
    sorted_countries = sorted(COUNTRY_NAMES.items(), key=lambda x: x[1])
    country_labels = [f"{name} ({code})" for code, name in sorted_countries]
    country_codes = [code for code, name in sorted_countries]
    
    default_idx = country_codes.index('FRA') if 'FRA' in country_codes else 0
    country_idx = st.selectbox(
        "Home country",
        range(len(country_labels)),
        format_func=lambda i: country_labels[i],
        index=default_idx,
        help="Used for travel cost estimation and geographic scheduling")
    player_country = country_codes[country_idx]
    
    st.divider()
    
    # --- Planning window ---
    st.subheader("Planning Window")
    
    season_label = st.selectbox(
        "Surface season",
        list(SURFACE_SEASONS.keys()),
        index=1,  # Default to clay
        help="Select a surface season or define a custom range")
    
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
    
    calendar_year = st.number_input(
        "Calendar year", min_value=2015, max_value=2025,
        value=2024,
        help="Which year's tournament calendar to use as template")
    
    st.divider()
    
    # --- Preferences ---
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
    
    max_budget = st.number_input(
        "Season budget ($)", min_value=0, max_value=100000,
        value=0, step=1000,
        help="Maximum total spend (travel + entry + accommodation). 0 = no limit")
    if max_budget == 0:
        max_budget = None
    
    st.divider()
    
    # --- Simulation settings ---
    st.subheader("Simulation")
    
    speed_mode = st.radio(
        "Speed vs accuracy",
        ['Fast (1-2 min)', 'Balanced (3-5 min)', 'Thorough (5-10 min)'],
        index=0,
        help="More simulations = more accurate but slower")
    
    sim_settings = {
        'Fast (1-2 min)': (150, 300, 1000),
        'Balanced (3-5 min)': (300, 500, 3000),
        'Thorough (5-10 min)': (500, 1000, 5000),
    }
    n_schedules, n_sims_tournament, n_sims_schedule = sim_settings[speed_mode]
    
    st.divider()
    
    # --- Run button ---
    run_clicked = st.button(
        "🚀 Optimize Schedule",
        use_container_width=True,
        type="primary")


# ==============================================================================
# MAIN AREA
# ==============================================================================

# Header
st.title("🎾 Seedr — Tournament Schedule Optimizer")
st.markdown(
    f"**{tour_label}** | Rank **{player_rank}** | "
    f"**{player_points}** points | "
    f"{COUNTRY_NAMES.get(player_country, player_country)} | "
    f"{season_label}")

# Check data availability
data_path = get_data_path(tour_code)
if not os.path.exists(data_path):
    st.error(
        f"Processed data not found at `{data_path}`. "
        f"Run the pipeline first: `python src/modeling/00_unified_pipeline.py`")
    st.stop()

# ==============================================================================
# RUN OPTIMIZER
# ==============================================================================

if run_clicked:
    with st.spinner("Loading optimizer and calendar data..."):
        optimizer = load_optimizer(tour_code, player_country, calendar_year)
    
    if optimizer is None:
        st.error("Failed to load optimizer. Check that processed data exists.")
        st.stop()
    
    # Update player country if changed
    if optimizer.player_country != player_country:
        st.cache_resource.clear()
        optimizer = load_optimizer(tour_code, player_country, calendar_year)
    
    st.info(f"Running optimizer: {n_schedules} candidate schedules × "
            f"{n_sims_schedule} simulations each...")
    
    progress_bar = st.progress(0, text="Initializing...")
    
    # Capture optimizer output
    t_start = time.time()
    
    progress_bar.progress(10, text="Computing per-tournament expected values...")
    
    results = optimizer.optimize(
        player_rank=player_rank,
        player_points=player_points,
        planning_start_week=start_week,
        planning_end_week=end_week,
        n_schedules=n_schedules,
        n_sims_per_tournament=n_sims_tournament,
        n_sims_per_schedule=n_sims_schedule,
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
    
    # Store results in session state
    st.session_state['results'] = results
    st.session_state['elapsed'] = elapsed

# ==============================================================================
# DISPLAY RESULTS
# ==============================================================================

if 'results' in st.session_state:
    results = st.session_state['results']
    elapsed = st.session_state.get('elapsed', 0)
    
    if 'error' in results:
        st.error(f"Optimization failed: {results['error']}")
        st.stop()
    
    meta = results['metadata']
    top_schedules = results['top_schedules']
    tournament_details = results.get('tournament_details', {})
    tournament_evs = results.get('tournament_evs', {})
    tournament_accept = results.get('tournament_accept', {})
    
    # --- Summary metrics ---
    st.success(
        f"Optimization complete in {elapsed:.0f}s — "
        f"{meta['n_eligible']} eligible tournaments, "
        f"{meta['n_schedules_generated']} schedules evaluated")
    
    best = top_schedules[0]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Expected Points", f"{best['expected_points']:.0f}",
                   help="Mean total ranking points across all simulations")
    with col2:
        st.metric("Expected Prize", format_currency(best['expected_prize']),
                   help="Mean total prize money")
    with col3:
        st.metric("Total Cost", format_currency(best['total_cost']),
                   help="Travel + entry fees + accommodation")
    with col4:
        st.metric("Net ROI", format_currency(best['net_prize']),
                   delta=format_currency(best['net_prize']),
                   delta_color="normal")
    with col5:
        rank_change = best['expected_final_rank'] - meta['player_rank']
        st.metric("Projected Rank",
                   f"{best['expected_final_rank']:.0f}",
                   delta=f"{rank_change:+.0f}",
                   delta_color="inverse")
    
    st.divider()
    
    # --- Schedule tabs ---
    if len(top_schedules) > 1:
        tab_labels = [f"Schedule {i+1}" for i in range(len(top_schedules))]
        tabs = st.tabs(tab_labels)
    else:
        tabs = [st.container()]
    
    for idx, (tab, sched) in enumerate(zip(tabs, top_schedules)):
        with tab:
            # Schedule overview metrics
            c1, c2, c3, c4 = tab.columns(4)
            with c1:
                st.metric("Points",
                           f"{sched['expected_points']:.0f}",
                           help=f"80% CI: [{sched['points_p10']:.0f} – {sched['points_p90']:.0f}]")
            with c2:
                st.metric("Prize Money", format_currency(sched['expected_prize']))
            with c3:
                st.metric("Total Cost", format_currency(sched['total_cost']))
            with c4:
                st.metric("Projected Rank",
                           f"{sched['expected_final_rank']:.0f}",
                           delta=f"{sched['expected_final_rank'] - meta['player_rank']:+.0f}",
                           delta_color="inverse")
            
            # Confidence intervals
            with st.expander("Confidence intervals"):
                ci_col1, ci_col2 = st.columns(2)
                with ci_col1:
                    st.markdown("**Points distribution:**")
                    st.markdown(
                        f"- Bad stretch (10th %ile): **{sched['points_p10']:.0f}**\n"
                        f"- Conservative (20th %ile): **{sched['points_p20']:.0f}**\n"
                        f"- Median: **{sched['points_p50']:.0f}**\n"
                        f"- Good stretch (80th %ile): **{sched['points_p80']:.0f}**\n"
                        f"- Great stretch (90th %ile): **{sched['points_p90']:.0f}**")
                with ci_col2:
                    st.markdown("**Rank projection:**")
                    st.markdown(
                        f"- Best case (10th %ile): **{sched['final_rank_p10']:.0f}**\n"
                        f"- Good (20th %ile): **{sched['final_rank_p20']:.0f}**\n"
                        f"- Expected: **{sched['expected_final_rank']:.0f}**\n"
                        f"- Rough (80th %ile): **{sched['final_rank_p80']:.0f}**\n"
                        f"- Worst case (90th %ile): **{sched['final_rank_p90']:.0f}**")
            
            # Cost breakdown
            with st.expander("Cost breakdown"):
                cost_data = {
                    'Category': ['Travel', 'Entry Fees', 'Accommodation', 'Total'],
                    'Cost': [
                        format_currency(sched['travel_cost']),
                        format_currency(sched['entry_fees']),
                        format_currency(sched['accommodation_cost']),
                        f"**{format_currency(sched['total_cost'])}**",
                    ]
                }
                st.table(pd.DataFrame(cost_data).set_index('Category'))
            
            st.markdown("---")
            st.markdown(f"**Tournament Schedule ({sched['n_tournaments']} events)**")
            
            # Build tournament table
            rows = []
            for week, tournament in sched['schedule']:
                name = tournament.get('tournament_name', '?')
                category = tournament.get('category', '?')
                surface = tournament.get('surface', '?')
                country = tournament.get('country', '?')
                ev = tournament_evs.get(name, 0)
                accept = tournament_accept.get(name, 1.0)
                
                # Get round probabilities
                details = tournament_details.get(name, {})
                round_probs = details.get('round_probs', {})
                exp_pts = details.get('expected_points', 0)
                exp_prize = details.get('expected_prize', 0)
                
                rows.append({
                    'Week': week,
                    'Tournament': name,
                    'Category': category,
                    'Surface': surface_badge(surface),
                    'Country': COUNTRY_NAMES.get(country, country) if isinstance(country, str) else '?',
                    'EV (pts)': f"{ev:.1f}",
                    'Exp. Prize': format_currency(exp_prize),
                    'Acceptance': f"{accept:.0%}",
                })
            
            df_schedule = pd.DataFrame(rows)
            st.dataframe(
                df_schedule,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Week': st.column_config.NumberColumn(width="small"),
                    'Tournament': st.column_config.TextColumn(width="large"),
                    'Category': st.column_config.TextColumn(width="medium"),
                    'EV (pts)': st.column_config.TextColumn(width="small"),
                })
            
            # Per-tournament round probabilities
            with st.expander("Round-by-round probabilities"):
                for week, tournament in sched['schedule']:
                    name = tournament.get('tournament_name', '?')
                    details = tournament_details.get(name, {})
                    round_probs = details.get('round_probs', {})
                    
                    if round_probs:
                        prob_str = " → ".join(
                            f"**{r}** {p:.0%}" for r, p in round_probs.items())
                        st.markdown(f"**{name}:** {prob_str}")
    
    st.divider()
    
    # --- All eligible tournaments ---
    with st.expander("📊 All eligible tournaments ranked by expected value"):
        all_rows = []
        for name, ev in sorted(tournament_evs.items(), key=lambda x: -x[1]):
            details = tournament_details.get(name, {})
            accept = tournament_accept.get(name, 1.0)
            raw_ev = results.get('tournament_raw_evs', {}).get(name, ev)
            
            all_rows.append({
                'Tournament': name,
                'Effective EV': f"{ev:.1f}",
                'Raw EV': f"{raw_ev:.1f}",
                'Acceptance': f"{accept:.0%}",
                'Exp. Prize': format_currency(details.get('expected_prize', 0)),
            })
        
        st.dataframe(
            pd.DataFrame(all_rows),
            use_container_width=True,
            hide_index=True)

else:
    # --- Welcome screen ---
    st.markdown("---")
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("""
        ### How it works
        
        1. **Set your profile** — Enter your ranking, points, and home country in the sidebar
        2. **Choose a planning window** — Pick a surface season or custom date range
        3. **Set preferences** — Surface, travel scope, and budget constraints
        4. **Hit Optimize** — The engine generates hundreds of candidate schedules, 
           simulates each one thousands of times with Monte Carlo methods, and ranks 
           them by expected ranking points
        5. **Compare options** — Review the top diverse schedules with full financial 
           projections and round-by-round probabilities
        
        ### What you get
        
        For each recommended schedule, Seedr shows you the expected ranking points 
        and prize money with confidence intervals, projected ranking at the end of the 
        window, total costs (travel, entry fees, accommodation), net financial return, 
        and per-tournament round-reach probabilities and acceptance chances.
        """)
    
    with col_right:
        st.markdown("""
        ### Quick start
        
        **Rank ~250 clay specialist:**
        - Rank: 250, Clay Season
        - Surface: Follow season
        - Travel: Continental
        
        **Rank ~500 ITF grinder:**
        - Rank: 500, any season
        - Surface: No preference
        - Budget: $5,000
        
        **Rank ~100 Slam qualifier:**
        - Rank: 100, Clay Season
        - Surface: Follow season
        - Travel: Global
        """)
    
    st.info("👈 Configure your profile in the sidebar and click **Optimize Schedule** to get started.")
