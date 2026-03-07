"""
Tennis Tournament Optimizer - Unified Pipeline (Both-Ranked Only)
Filters to matches where both players have rankings, then applies
score parsing, dates, tiers, and demographics.
"""
import pandas as pd
import numpy as np
import re
import time
import gc
import os
import sys

# Add modeling directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from birth_dates import BirthDateLookup

# ==============================================================================
# TIER MAPS
# ==============================================================================
ATP_TIERS = {
    "Grand Slam (Men's)":(1,"Grand Slam","pro","ranking"),
    "ATP 1000":(2,"ATP 1000","pro","commitment_top30"),
    "ATP 500":(3,"ATP 500","pro","semi_top30"),
    "ATP 250":(4,"ATP 250","pro","optional"),
    "ATP Finals":(4,"ATP Finals","pro","qualification"),
    "Challengers":(5,"Challenger","pro","optional"),
    "Challenger 50":(5,"Challenger","pro","optional"),
    "Challenger 75":(5,"Challenger","pro","optional"),
    "Challenger 80":(5,"Challenger","pro","optional"),
    "Challenger 90":(5,"Challenger","pro","optional"),
    "Challenger 100":(5,"Challenger","pro","optional"),
    "Challenger 110":(5,"Challenger","pro","optional"),
    "Challenger 125":(5,"Challenger","pro","optional"),
    "Challenger 175":(5,"Challenger","pro","optional"),
    "Challenger Tour Finals":(5,"Challenger","pro","qualification"),
    "M25":(6,"ITF M25","pro","optional"),
    "M15":(6,"ITF M15","pro","optional"),
    "Davis Cup World Group":(7,"Team/Exhibition","pro","optional"),
    "Davis Cup":(7,"Team/Exhibition","pro","optional"),
    "Olympic Games (Men's)":(7,"Team/Exhibition","pro","optional"),
    "JR Grand Slam Boys":(8,"Jr Grand Slam","junior","optional"),
    "JR Masters Boys":(8,"Jr Grand Slam","junior","qualification"),
    "J500 Boys":(9,"Junior High","junior","optional"),
    "J300 Boys":(9,"Junior High","junior","optional"),
    "J200 Boys":(9,"Junior High","junior","optional"),
    "JB1 Boys":(9,"Junior High","junior","optional"),
    "JB2 Boys":(9,"Junior High","junior","optional"),
    "Youth Olympics Boys":(9,"Junior High","junior","optional"),
    "J100 Boys":(10,"Junior Mid","junior","optional"),
    "J60 Boys":(10,"Junior Mid","junior","optional"),
    "JB3 Boys":(10,"Junior Mid","junior","optional"),
    "J30 Boys":(10,"Junior Low","junior","optional"),
    "Boys 18":(11,"Youth 18U","youth","optional"),
    "Boys 16":(11,"Youth 16U","youth","optional"),
    "Boys 16 Cat 1":(11,"Youth 16U","youth","optional"),
    "Boys 16 Cat 2":(11,"Youth 16U","youth","optional"),
    "Boys 16 Cat 3":(11,"Youth 16U","youth","optional"),
    "Boys 16 Masters":(11,"Youth 16U","youth","optional"),
    "Boys 16 European Championship":(11,"Youth 16U","youth","optional"),
    "Boys 16 Super Cat":(11,"Youth 16U","youth","optional"),
    "Boys 14":(12,"Youth 14U","youth","optional"),
    "Boys 14 Cat 1":(12,"Youth 14U","youth","optional"),
    "Boys 14 Cat 2":(12,"Youth 14U","youth","optional"),
    "Boys 14 Cat 3":(12,"Youth 14U","youth","optional"),
    "Boys 14 Masters":(12,"Youth 14U","youth","optional"),
    "Boys 14 European Championship":(12,"Youth 14U","youth","optional"),
    "Boys 14 Super Cat":(12,"Youth 14U","youth","optional"),
    "Boys 12":(13,"Youth 12U","youth","optional"),
    "Boys 12 Cat 1":(13,"Youth 12U","youth","optional"),
    "Boys 12 Cat 2":(13,"Youth 12U","youth","optional"),
    "Boys 12 Super Cat":(13,"Youth 12U","youth","optional"),
}

WTA_TIERS = {
    "Grand Slam (Women's)":(1,"Grand Slam","pro","ranking"),
    "WTA 1000":(2,"WTA 1000","pro","commitment_top30"),
    "WTA 1000 (5)":(2,"WTA 1000","pro","commitment_top30"),
    "WTA 500":(3,"WTA 500","pro","semi_top30"),
    "WTA 250":(4,"WTA 250","pro","optional"),
    "WTA 125":(4,"WTA 125","pro","optional"),
    "Year End Championships":(4,"WTA Finals","pro","qualification"),
    "Tier III":(4,"WTA 250","pro","optional"),
    "W100":(5,"ITF W100","pro","optional"),
    "W80":(5,"ITF W80","pro","optional"),
    "W75":(5,"ITF W75","pro","optional"),
    "W50":(5,"ITF W50","pro","optional"),
    "W35":(6,"ITF W35","pro","optional"),
    "W15":(6,"ITF W15","pro","optional"),
    "W10":(6,"ITF W10","pro","optional"),
    "Fed Cup World Group":(7,"Team/Exhibition","pro","optional"),
    "Fed Cup World Group 2":(7,"Team/Exhibition","pro","optional"),
    "Fed Cup":(7,"Team/Exhibition","pro","optional"),
    "Olympic Games (Women's)":(7,"Team/Exhibition","pro","optional"),
    "JR Grand Slam Girls":(8,"Jr Grand Slam","junior","optional"),
    "JR Masters Girls":(8,"Jr Grand Slam","junior","qualification"),
    "J500 Girls":(9,"Junior High","junior","optional"),
    "J300 Girls":(9,"Junior High","junior","optional"),
    "J200 Girls":(9,"Junior High","junior","optional"),
    "JB1 Girls":(9,"Junior High","junior","optional"),
    "JB2 Girls":(9,"Junior High","junior","optional"),
    "Youth Olympics Girls":(9,"Junior High","junior","optional"),
    "J100 Girls":(10,"Junior Mid","junior","optional"),
    "J60 Girls":(10,"Junior Mid","junior","optional"),
    "JB3 Girls":(10,"Junior Mid","junior","optional"),
    "J30 Girls":(10,"Junior Low","junior","optional"),
    "J30 Boys":(10,"Junior Low","junior","optional"),
    "Girls 18":(11,"Youth 18U","youth","optional"),
    "Girls 16":(11,"Youth 16U","youth","optional"),
    "Girls 16 Cat 1":(11,"Youth 16U","youth","optional"),
    "Girls 16 Cat 2":(11,"Youth 16U","youth","optional"),
    "Girls 16 Cat 3":(11,"Youth 16U","youth","optional"),
    "Girls 16 Masters":(11,"Youth 16U","youth","optional"),
    "Girls 16 European Championship":(11,"Youth 16U","youth","optional"),
    "Girls 16 Super Cat":(11,"Youth 16U","youth","optional"),
    "Girls 14":(12,"Youth 14U","youth","optional"),
    "Girls 14 Cat 1":(12,"Youth 14U","youth","optional"),
    "Girls 14 Cat 2":(12,"Youth 14U","youth","optional"),
    "Girls 14 Cat 3":(12,"Youth 14U","youth","optional"),
    "Girls 14 Masters":(12,"Youth 14U","youth","optional"),
    "Girls 14 European Championship":(12,"Youth 14U","youth","optional"),
    "Girls 14 Super Cat":(12,"Youth 14U","youth","optional"),
    "Girls 12":(13,"Youth 12U","youth","optional"),
    "Girls 12 Cat 1":(13,"Youth 12U","youth","optional"),
    "Girls 12 Cat 2":(13,"Youth 12U","youth","optional"),
    "Girls 12 Super Cat":(13,"Youth 12U","youth","optional"),
}


# ==============================================================================
# SCORE PARSING (vectorized)
# ==============================================================================
def parse_scores(df):
    print("    Parsing scores...")
    t0 = time.time()
    score = df['score'].fillna('').str.strip()

    # Match status
    ms = pd.Series('completed', index=df.index)
    ms[score == ''] = 'missing'
    ms[score.str.lower() == 'w/o'] = 'walkover'
    ms[score.str.lower() == 'def.'] = 'default_bare'
    ms[score.str.contains("Ret'd", case=False, na=False) & (ms == 'completed')] = 'retirement'
    ms[score.str.contains(r"Def\.", case=False, regex=True, na=False) & (ms == 'completed')] = 'default'

    # Clean and split
    sc = score.str.replace("Ret'd", "", case=False, regex=False)
    sc = sc.str.replace("Def.", "", case=False, regex=False).str.strip()
    sets_list = sc.str.split()
    n_sets = sets_list.apply(lambda x: len([s for s in x if s]) if isinstance(x, list) else 0)
    n_sets[ms.isin(['missing', 'walkover', 'default_bare'])] = np.nan

    # Pad to 5 sets
    def pad5(x):
        if not isinstance(x, list): x = []
        x = [s for s in x if s]
        return (x + [None] * 5)[:5]

    padded = sets_list.apply(pad5)
    set_strs = pd.DataFrame(padded.tolist(), index=df.index, columns=[f's{i}' for i in range(5)])

    # Vectorized per-set parsing
    has_match_tb = pd.Series(False, index=df.index)
    all_w = []; all_l = []

    for i in range(5):
        col = set_strs[f's{i}'].fillna('')
        w = pd.Series(np.nan, index=df.index)
        l = pd.Series(np.nan, index=df.index)

        tb = col.str.match(r'^\d\d\(\d+\)$')
        if tb.any():
            w[tb] = col[tb].str[0].astype(float)
            l[tb] = col[tb].str[1].astype(float)

        std = col.str.match(r'^\d\d$') & ~tb
        if std.any():
            w[std] = col[std].str[0].astype(float)
            l[std] = col[std].str[1].astype(float)

        m3 = col.str.match(r'^\d{3}$') & ~tb & ~std
        if m3.any():
            w[m3] = col[m3].str[:2].astype(float)
            l[m3] = col[m3].str[2:].astype(float)
            has_match_tb = has_match_tb | m3

        m4 = col.str.match(r'^\d{4}$') & ~tb & ~std & ~m3
        if m4.any():
            w[m4] = col[m4].str[:2].astype(float)
            l[m4] = col[m4].str[2:].astype(float)
            has_match_tb = has_match_tb | m4

        all_w.append(w); all_l.append(l)
        df[f'set{i+1}_w'] = w; df[f'set{i+1}_l'] = l

    # Aggregate
    w_games = sum(x.fillna(0) for x in all_w)
    l_games = sum(x.fillna(0) for x in all_l)
    w_sets = sum((all_w[i] > all_l[i]).fillna(False).astype(int) for i in range(5))
    l_sets = sum((all_l[i] > all_w[i]).fillna(False).astype(int) for i in range(5))

    nos = ms.isin(['missing', 'walkover', 'default_bare'])
    w_games[nos] = np.nan; l_games[nos] = np.nan
    w_sets[nos] = np.nan; l_sets[nos] = np.nan

    # Player perspective
    is_win = df['result'] == 'W'
    p_sets_won = np.where(is_win, w_sets, l_sets)
    p_sets_lost = np.where(is_win, l_sets, w_sets)
    p_games_won = np.where(is_win, w_games, l_games)
    p_games_lost = np.where(is_win, l_games, w_games)

    # Score validity
    score_valid = pd.Series(True, index=df.index)
    comp = ms == 'completed'
    score_valid[comp & pd.notna(w_sets) & pd.notna(l_sets) & (w_sets <= l_sets)] = False

    df['match_status'] = ms
    df['n_sets_played'] = n_sets
    df['has_match_tb'] = has_match_tb
    df['score_valid'] = score_valid
    df['w_sets'] = w_sets; df['l_sets'] = l_sets
    df['w_games_total'] = w_games; df['l_games_total'] = l_games
    df['p_sets_won'] = p_sets_won; df['p_sets_lost'] = p_sets_lost
    df['p_games_won'] = p_games_won; df['p_games_lost'] = p_games_lost

    print(f"      Done ({time.time()-t0:.1f}s)")
    return df


# ==============================================================================
# RANKINGS JOIN (merge_asof)
# ==============================================================================
def join_rankings_for_role(df, rankings_sorted, id_col, date_col, prefix):
    """Join rankings for player or opponent using merge_asof."""
    merged = df[[id_col, date_col]].copy()
    merged.columns = ['pid', 'mdate']
    merged = merged.sort_values('mdate').reset_index()

    result = pd.merge_asof(
        merged,
        rankings_sorted,
        left_on='mdate', right_on='week_date',
        by='pid',
        direction='backward',
        tolerance=pd.Timedelta(days=60)
    )
    result = result.sort_values('index').set_index('index')

    df[f'{prefix}_rank'] = result['ranking'].values
    df[f'{prefix}_points'] = result['points'].values
    return df


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def process_tour(base_path, tour, tier_map):
    label = tour.upper()
    t_start = time.time()
    print(f"\n{'='*60}")
    print(f" {label} — Both-Ranked Pipeline")
    print(f"{'='*60}\n")

    raw_path = f"{base_path}/data/raw/{tour}"
    out_path = f"{base_path}/data/processed"
    os.makedirs(out_path, exist_ok=True)

    # =========================================================================
    # STEP 1: Load matches and parse dates (needed for ranking join)
    # =========================================================================
    print("  [1/5] Loading matches and parsing dates...")
    t0 = time.time()
    match_file = f"{raw_path}/{tour}_all_matches.csv"
    if not os.path.exists(match_file):
        match_file += ".gz"
    df = pd.read_csv(match_file)
    n_total = len(df)
    print(f"    {n_total:,} total rows ({time.time()-t0:.0f}s)")

    df['start_date_parsed'] = pd.to_datetime(
        df['start_date'] + ' ' + df['year'].astype(str),
        format='%b %d %Y', errors='coerce')
    df['end_date_parsed'] = pd.to_datetime(
        df['end_date'] + ' ' + df['year'].astype(str),
        format='%b %d %Y', errors='coerce')

    # Fix year boundary
    wrap = df['end_date_parsed'].notna() & df['start_date_parsed'].notna() & \
           (df['end_date_parsed'] < df['start_date_parsed'])
    if wrap.any():
        df.loc[wrap, 'end_date_parsed'] = pd.to_datetime(
            df.loc[wrap, 'end_date'] + ' ' + (df.loc[wrap, 'year'] + 1).astype(str),
            format='%b %d %Y', errors='coerce')
        print(f"    Fixed {wrap.sum()} year-boundary dates")

    df['start_week_monday'] = df['start_date_parsed'] - pd.to_timedelta(
        df['start_date_parsed'].dt.dayofweek, unit='D')

    # =========================================================================
    # STEP 2: Join rankings (both pro and junior)
    # =========================================================================
    print("\n  [2/5] Joining rankings...")
    t0 = time.time()
    rank_file = f"{raw_path}/{tour}_all_weekly_rankings.csv"
    if not os.path.exists(rank_file):
        rank_file += ".gz"
    rankings = pd.read_csv(rank_file)
    rankings['week_date'] = pd.to_datetime(rankings['week_date'])

    pro_type = 'ATP' if tour == 'atp' else 'WTA'
    junior_type = 'ITF Junior'

    for rank_type, prefix_label in [(pro_type, 'pro'), (junior_type, 'junior')]:
        rt = rankings[rankings['ranking_type'] == rank_type][
            ['player_id', 'week_date', 'ranking', 'points']].copy()
        rt = rt.rename(columns={'player_id': 'pid'}).sort_values('week_date')

        if len(rt) == 0:
            for role in ['player', 'opponent']:
                df[f'{role}_{prefix_label}_rank'] = np.nan
                df[f'{role}_{prefix_label}_points'] = np.nan
            continue

        # Player rankings
        df = join_rankings_for_role(df, rt, 'player_id', 'start_date_parsed',
                                    f'player_{prefix_label}')
        # Opponent rankings
        df = join_rankings_for_role(df, rt, 'opponent_id', 'start_date_parsed',
                                    f'opponent_{prefix_label}')

    del rankings; gc.collect()

    # Combine: prefer pro, fall back to junior
    for role in ['player', 'opponent']:
        df[f'{role}_rank'] = df[f'{role}_pro_rank'].fillna(df[f'{role}_junior_rank'])
        df[f'{role}_points'] = df[f'{role}_pro_points'].fillna(df[f'{role}_junior_points'])
        df[f'{role}_rank_type'] = np.where(
            df[f'{role}_pro_rank'].notna(), 'pro',
            np.where(df[f'{role}_junior_rank'].notna(), 'junior', None))

    df['rank_diff'] = df['opponent_rank'] - df['player_rank']
    df['points_diff'] = df['player_points'] - df['opponent_points']

    # Ranking match type: both_pro, both_junior, mixed, or incomplete
    df['ranking_match_type'] = np.where(
        df['player_rank'].isna() | df['opponent_rank'].isna(), 'incomplete',
        np.where(
            (df['player_rank_type'] == 'pro') & (df['opponent_rank_type'] == 'pro'), 'both_pro',
            np.where(
                (df['player_rank_type'] == 'junior') & (df['opponent_rank_type'] == 'junior'), 'both_junior',
                'mixed')))

    both_ranked = df['player_rank'].notna() & df['opponent_rank'].notna()
    print(f"    Rankings joined ({time.time()-t0:.0f}s)")
    print(f"    Both-ranked: {both_ranked.sum():,} / {n_total:,} ({100*both_ranked.mean():.1f}%)")
    print(f"    Ranking match types:")
    for rmt, count in df['ranking_match_type'].value_counts().items():
        print(f"      {rmt:<15s}: {count:>10,} ({100*count/n_total:.1f}%)")

    # =========================================================================
    # STEP 3: Filter to both-ranked only
    # =========================================================================
    print(f"\n  [3/5] Filtering to both-ranked matches...")
    df = df[both_ranked].copy().reset_index(drop=True)
    print(f"    Kept {len(df):,} rows")
    gc.collect()

    # =========================================================================
    # STEP 4: Score parsing
    # =========================================================================
    print("\n  [4/5] Score parsing...")
    df = parse_scores(df)
    print(f"    Status: {df['match_status'].value_counts().to_dict()}")
    print(f"    Score quality issues: {(~df['score_valid']).sum()}")

    # =========================================================================
    # STEP 5: Tiers + Demographics
    # =========================================================================
    print("\n  [5/5] Tiers and demographics...")

    # Tiers
    tier_df = pd.DataFrame([
        {'category': k, 'tier': v[0], 'tier_name': v[1], 'level': v[2], 'mandatory': v[3]}
        for k, v in tier_map.items()
    ])
    df = df.merge(tier_df, on='category', how='left')
    unmapped = df['tier'].isna().sum()
    if unmapped > 0:
        print(f"    WARNING: {unmapped} unmapped categories")
        print(f"    {df[df['tier'].isna()]['category'].value_counts().head().to_dict()}")

    # Demographics (using precise birth dates from rankings, fallback to birth year)
    rank_file_for_birth = f"{raw_path}/{tour}_all_weekly_rankings.csv"
    if not os.path.exists(rank_file_for_birth):
        rank_file_for_birth += ".gz"
    profile_file = f"{raw_path}/{tour}_player_profiles.csv"
    if not os.path.exists(profile_file):
        profile_file += ".gz"

    birth_lookup = BirthDateLookup()
    birth_lookup.load(rank_file_for_birth, profile_file)

    df['player_age'] = birth_lookup.compute_ages_fast(df, 'player_id', 'start_date_parsed')
    df['opponent_age'] = birth_lookup.compute_ages_fast(df, 'opponent_id', 'start_date_parsed')

    player_age_coverage = df['player_age'].notna().mean()
    opponent_age_coverage = df['opponent_age'].notna().mean()
    print(f"    Age coverage: player={player_age_coverage:.0%}, opponent={opponent_age_coverage:.0%}")

    # =========================================================================
    # REORDER COLUMNS
    # =========================================================================
    col_order = [
        "player_id","opponent_id","opponent_name","opponent_country",
        "tournament_name","tournament_url","year","start_date","end_date",
        "start_date_parsed","end_date_parsed","start_week_monday",
        "location","country","surface",
        "category","tier","tier_name","level","mandatory",
        "is_qualifying","round",
        "result","score","match_status","score_valid",
        "n_sets_played","has_match_tb",
        "w_sets","l_sets","w_games_total","l_games_total",
        "p_sets_won","p_sets_lost","p_games_won","p_games_lost",
        "set1_w","set1_l","set2_w","set2_l","set3_w","set3_l",
        "set4_w","set4_l","set5_w","set5_l",
        "player_rank","player_points","player_rank_type",
        "opponent_rank","opponent_points","opponent_rank_type",
        "rank_diff","points_diff","ranking_match_type",
        "player_pro_rank","player_pro_points",
        "player_junior_rank","player_junior_points",
        "opponent_pro_rank","opponent_pro_points",
        "opponent_junior_rank","opponent_junior_points",
        "player_age","opponent_age",
    ]
    existing = [c for c in col_order if c in df.columns]
    remaining = [c for c in df.columns if c not in col_order]
    df = df[existing + remaining]

    # =========================================================================
    # VALIDATION
    # =========================================================================
    print(f"\n  --- Validation ---")
    print(f"    Total rows: {len(df):,}")
    print(f"    Columns: {len(df.columns)}")

    # Ranking match type breakdown
    print(f"\n    Ranking match types:")
    for rmt, count in df['ranking_match_type'].value_counts().items():
        print(f"      {rmt:<15s}: {count:>10,} ({100*count/len(df):.1f}%)")

    # Win rate monotonicity (both_pro only for clean signal)
    both_pro = df[(df['result'].isin(['W', 'L'])) & 
                  (df['ranking_match_type'] == 'both_pro') &
                  (df['rank_diff'].notna())].copy()
    both_pro['bucket'] = pd.cut(both_pro['rank_diff'],
                             bins=[-np.inf, -500, -200, -50, 0, 50, 200, 500, np.inf])
    wr = both_pro.groupby('bucket', observed=False)['result'].apply(lambda x: (x == 'W').mean())
    cn = both_pro.groupby('bucket', observed=False)['result'].count()

    print(f"\n    Win rate by rank advantage (both_pro only, n={len(both_pro):,}):")
    for b, w in wr.items():
        print(f"      {str(b):>20s}: {100*w:.1f}%  (n={cn[b]:,})")

    is_mono = all(x > 0 for x in wr.diff().dropna())
    print(f"    Monotonic: {'YES' if is_mono else 'NO'}")

    # Tier distribution
    print(f"\n    Tier distribution:")
    tier_counts = df.groupby(['tier', 'tier_name'], observed=False).size()
    for (t, tn), c in tier_counts.items():
        if pd.notna(t):
            print(f"      T{int(t):>2d} {tn:<20s} {c:>10,}")

    # Level split
    print(f"\n    Level: {df['level'].value_counts().to_dict()}")
    print(f"    Surfaces: {df['surface'].value_counts().head(5).to_dict()}")

    elapsed = time.time() - t_start
    print(f"\n  DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return df


# ==============================================================================
# RUN
# ==============================================================================
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

# ATP
atp = process_tour(BASE, "atp", ATP_TIERS)
out_file = f"{BASE}/data/processed/atp_clean_both_ranked.csv"
atp.to_csv(out_file, index=False)
size_mb = os.path.getsize(out_file) / 1024**2
print(f"\n  Saved {out_file} ({size_mb:.0f} MB)")

del atp; gc.collect()

# WTA
wta = process_tour(BASE, "wta", WTA_TIERS)
out_file = f"{BASE}/data/processed/wta_clean_both_ranked.csv"
wta.to_csv(out_file, index=False)
size_mb = os.path.getsize(out_file) / 1024**2
print(f"\n  Saved {out_file} ({size_mb:.0f} MB)")

print(f"\n{'='*60}")
print("PIPELINE COMPLETE")
print(f"{'='*60}")
