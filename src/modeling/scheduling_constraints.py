"""
Tennis Tournament Optimizer - Rest & Scheduling Constraints
===========================================================

Empirical rest patterns derived from 340,912 player-tournament entries
(ATP pro level, 2015-2026). Used by the seasonal optimizer to set
realistic scheduling constraints.
"""

# ==============================================================================
# REST PARAMETERS (days between tournament end and next start)
# ==============================================================================

# Minimum rest by tournament tier just completed (empirical P10)
MIN_REST_DAYS = {
    "Grand Slam": 7,       # Best-of-5, two-week events demand recovery
    "Masters 1000": 1,
    "ATP 500": 1,
    "ATP 250": 1,
    "Challenger": 1,
    "ITF": 1,
}

# Recommended rest by tournament tier (empirical median)
RECOMMENDED_REST_DAYS = {
    "Grand Slam": 8,
    "Masters 1000": 5,
    "ATP 500": 4,
    "ATP 250": 4,
    "Challenger": 6,
    "ITF": 8,
}

# ==============================================================================
# SEASON PARAMETERS BY RANK LEVEL
# ==============================================================================

# Typical tournaments per year (median, P25, P75)
TOURNAMENTS_PER_YEAR = {
    "top_10":   {"median": 16, "p25": 10, "p75": 21},
    "top_30":   {"median": 21, "p25":  9, "p75": 24},
    "top_50":   {"median": 22, "p25":  5, "p75": 27},
    "51_100":   {"median": 22, "p25":  4, "p75": 28},
    "101_200":  {"median": 20, "p25":  4, "p75": 29},
    "201_500":  {"median": 17, "p25":  4, "p75": 27},
    "500_plus": {"median":  7, "p25":  3, "p75": 15},
}

# Season length in weeks (median)
SEASON_LENGTH_WEEKS = {
    "top_10": 43, "top_30": 42, "top_50": 43,
    "51_100": 43, "101_200": 44, "201_500": 43, "500_plus": 38,
}

# Back-to-back rate (% of transitions with 0-1 day gap)
BACK_TO_BACK_RATE = {
    "top_10": 0.368, "top_30": 0.524, "top_50": 0.617,
    "51_100": 0.634, "101_200": 0.573, "201_500": 0.623, "500_plus": 0.598,
}

# ==============================================================================
# SCHEDULING CONSTRAINTS FOR OPTIMIZER
# ==============================================================================

def get_rank_bracket(rank):
    """Map a ranking to its bracket key."""
    if rank <= 10: return "top_10"
    if rank <= 30: return "top_30"
    if rank <= 50: return "top_50"
    if rank <= 100: return "51_100"
    if rank <= 200: return "101_200"
    if rank <= 500: return "201_500"
    return "500_plus"


def get_scheduling_constraints(player_rank, tier_just_played=None):
    """
    Get scheduling constraints for a player.
    
    Args:
        player_rank: Current ranking
        tier_just_played: Tier of tournament just completed (for rest calc)
    
    Returns:
        dict with constraint parameters
    """
    bracket = get_rank_bracket(player_rank)
    
    constraints = {
        # How many tournaments to plan for
        "target_tournaments": TOURNAMENTS_PER_YEAR[bracket]["median"],
        "max_tournaments": TOURNAMENTS_PER_YEAR[bracket]["p75"],
        "min_tournaments": max(5, TOURNAMENTS_PER_YEAR[bracket]["p25"]),
        
        # Season structure
        "season_weeks": SEASON_LENGTH_WEEKS[bracket],
        
        # Maximum consecutive back-to-back weeks before forced rest
        "max_consecutive_weeks": 4,
        
        # Minimum rest after current tournament
        "min_rest_days": MIN_REST_DAYS.get(tier_just_played, 1) if tier_just_played else 1,
        "recommended_rest_days": RECOMMENDED_REST_DAYS.get(tier_just_played, 7) if tier_just_played else 7,
        
        # Fatigue warning threshold
        "fatigue_warning_consecutive": 3,  # warn after 3 back-to-back
    }
    
    return constraints


def validate_schedule(tournament_dates, player_rank):
    """
    Validate a proposed tournament schedule against empirical patterns.
    
    Args:
        tournament_dates: List of (start_date, end_date, tier_name) tuples
        player_rank: Player's ranking
    
    Returns:
        dict with validation results and warnings
    """
    bracket = get_rank_bracket(player_rank)
    warnings = []
    
    n = len(tournament_dates)
    typical = TOURNAMENTS_PER_YEAR[bracket]
    
    # Check total count
    if n > typical["p75"] + 3:
        warnings.append(f"Very heavy schedule: {n} tournaments (typical max ~{typical['p75']})")
    elif n > typical["p75"]:
        warnings.append(f"Heavy schedule: {n} tournaments (typical P75={typical['p75']})")
    
    # Check gaps
    sorted_dates = sorted(tournament_dates, key=lambda x: x[0])
    consecutive_count = 0
    max_consecutive = 0
    
    for i in range(1, len(sorted_dates)):
        prev_end = sorted_dates[i-1][1]
        curr_start = sorted_dates[i][0]
        gap = (curr_start - prev_end).days
        prev_tier = sorted_dates[i-1][2]
        
        min_rest = MIN_REST_DAYS.get(prev_tier, 1)
        
        if gap < min_rest:
            warnings.append(
                f"Insufficient rest after {prev_tier}: {gap}d gap (min {min_rest}d)")
        
        if gap <= 1:
            consecutive_count += 1
            max_consecutive = max(max_consecutive, consecutive_count)
        else:
            consecutive_count = 0
    
    if max_consecutive >= 4:
        warnings.append(
            f"Long back-to-back stretch: {max_consecutive+1} consecutive weeks")
    
    return {
        "valid": len(warnings) == 0,
        "n_tournaments": n,
        "typical_range": f"{typical['p25']}-{typical['p75']}",
        "max_consecutive_weeks": max_consecutive + 1,
        "warnings": warnings,
    }


# ==============================================================================
# SEASONAL SURFACE MAP (empirical from 2022-2025 ATP pro data)
# ==============================================================================
# Maps week -> (dominant_surface, share)
# Simplified into seasonal blocks for clarity

def get_seasonal_surface(week):
    """
    Return the dominant surface and its share for a given calendar week.
    Based on actual ATP pro match distribution 2022-2025.
    
    Returns:
        (surface_name, share) e.g. ("Clay", 0.78)
    """
    if 13 <= week <= 21:      # Clay season (late March - late May)
        return ("Clay", 0.75)
    elif 22 <= week <= 26:    # Mixed clay/grass (late May - late June)
        return ("Clay", 0.55)  # Clay still dominant, some grass
    elif 27 <= week <= 34:    # Summer clay/hard (July - August)
        return ("Clay", 0.52)
    elif 35 <= week <= 39:    # US Open series / hard (late Aug - Sep)
        return ("Hard", 0.50)
    elif 40 <= week <= 46:    # Indoor hard season (Oct - mid Nov)
        return ("Hard Indoor", 0.35)  # Mixed hard/indoor
    else:                      # Off-season / Australian swing (Nov-Mar)
        return ("Hard", 0.60)


def get_surface_weight(tournament_surface, week, player_rank):
    """
    Compute a weighting multiplier (0-1) for a tournament surface
    given the current week and player rank.
    
    Higher-ranked players follow the season more strictly.
    Lower-ranked players are more flexible.
    
    Returns:
        float multiplier: 1.0 for seasonal match, down to 0.2 for off-season surface
    """
    seasonal_surface, share = get_seasonal_surface(week)
    
    # Normalize surface names for comparison
    # "Hard Indoor" matches both "Hard" and "Hard Indoor" tournaments
    t_surf = tournament_surface or ''
    
    def surfaces_match(surface, seasonal):
        if surface == seasonal:
            return True
        if seasonal == "Hard Indoor" and surface in ("Hard", "Hard Indoor"):
            return True
        if seasonal in ("Hard", "Clay") and surface == seasonal:
            return True
        return False
    
    is_match = surfaces_match(t_surf, seasonal_surface)
    
    # Rank-dependent strictness (top players follow season more)
    if player_rank <= 100:
        strictness = 0.9   # Very strong surface preference
    elif player_rank <= 300:
        strictness = 0.7   # Strong preference
    elif player_rank <= 600:
        strictness = 0.5   # Moderate preference
    else:
        strictness = 0.3   # Weaker preference (play what's available)
    
    if is_match:
        return 1.0
    else:
        # Penalty scales with strictness and how dominant the season is
        penalty = strictness * share  # e.g. 0.7 * 0.75 = 0.525 penalty
        return max(0.15, 1.0 - penalty)


# ==============================================================================
# DEMO
# ==============================================================================
if __name__ == '__main__':
    print("Rest & Scheduling Constraints")
    print("=" * 50)
    
    for rank in [10, 50, 100, 200, 500]:
        c = get_scheduling_constraints(rank, "Challenger")
        print(f"\n  Rank {rank}:")
        print(f"    Target tournaments: {c['target_tournaments']} "
              f"(range {c['min_tournaments']}-{c['max_tournaments']})")
        print(f"    Season: {c['season_weeks']} weeks")
        print(f"    Min rest after Challenger: {c['min_rest_days']}d")
        print(f"    Recommended rest: {c['recommended_rest_days']}d")
    
    # Post-Grand-Slam rest
    print(f"\n  After a Grand Slam:")
    c = get_scheduling_constraints(100, "Grand Slam")
    print(f"    Min rest: {c['min_rest_days']}d")
    print(f"    Recommended: {c['recommended_rest_days']}d")
