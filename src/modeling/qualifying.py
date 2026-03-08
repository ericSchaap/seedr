"""
Tennis Tournament Optimizer - Qualifying Pathway
==================================================

Models the qualifying draw as an alternative when a player is not
accepted into the main draw.

Structure (from ATP rules and data analysis):
  - Challengers: 24-player qualifying, ~2 rounds to win through
  - ATP 250/500: 16-32 player qualifying, 2-3 rounds
  - ATP 1000: 48-96 player qualifying, 2-3 rounds
  - Grand Slams: 128-player qualifying, 3 rounds
  - ITF (M15/M25): No separate qualifying (wildcards/alternates instead)

Qualifying field strength (from 2022-2025 data):
  - Challenger qualifying: median rank ~475 (vs main draw ~251)
  - ATP 250 qualifying: median rank ~175 (vs main draw ~80)
  - ATP 1000 qualifying: median rank ~115 (vs main draw ~50)

Key rules:
  - Challenger qualifying: no separate qualifying points
  - ATP 250/500: Q1=0, Q2=6-10, Q3=12-20 bonus points
  - Qualifiers enter main draw as the weakest unseeded players
  - Hotel accommodation covers qualifying players at Challengers

Usage:
    from qualifying import QualifyingPathway

    qp = QualifyingPathway()

    # Check if player can enter qualifying
    can_q = qp.can_enter_qualifying("Challenger 75", player_rank=400)

    # Simulate qualifying attempt
    result = qp.simulate_qualifying(
        player_rank=400, category="Challenger 75",
        surface="Clay", rng=rng, win_model=win_model)
    # Returns: {'qualified': True/False, 'rounds_played': 2,
    #           'qualifying_points': 0, 'qualifying_prize': 0}
"""

import random


# =========================================================================
# QUALIFYING DRAW STRUCTURE
# =========================================================================

QUALIFYING_STRUCTURE = {
    # category: {draw_size, n_rounds, n_qualifiers, has_bonus_points}
    "Grand Slam (Men's)":  {'draw': 128, 'rounds': 3, 'qualifiers': 16, 'bonus_points': True},
    "Grand Slam (Women's)":{'draw': 96,  'rounds': 3, 'qualifiers': 12, 'bonus_points': True},
    "ATP 1000":            {'draw': 48,  'rounds': 3, 'qualifiers': 8,  'bonus_points': True},
    "WTA 1000":            {'draw': 48,  'rounds': 3, 'qualifiers': 8,  'bonus_points': True},
    "ATP 500":             {'draw': 16,  'rounds': 2, 'qualifiers': 4,  'bonus_points': True},
    "WTA 500":             {'draw': 16,  'rounds': 2, 'qualifiers': 4,  'bonus_points': True},
    "ATP 250":             {'draw': 16,  'rounds': 2, 'qualifiers': 4,  'bonus_points': True},
    "WTA 250":             {'draw': 16,  'rounds': 2, 'qualifiers': 4,  'bonus_points': True},
    "Challenger 175":      {'draw': 24,  'rounds': 2, 'qualifiers': 4,  'bonus_points': False},
    "Challenger 125":      {'draw': 24,  'rounds': 2, 'qualifiers': 4,  'bonus_points': False},
    "Challenger 100":      {'draw': 24,  'rounds': 2, 'qualifiers': 4,  'bonus_points': False},
    "Challenger 75":       {'draw': 24,  'rounds': 2, 'qualifiers': 4,  'bonus_points': False},
    "Challenger 50":       {'draw': 24,  'rounds': 2, 'qualifiers': 4,  'bonus_points': False},
    # ITF events: no formal qualifying draw (alternates/wildcards only)
    "M25":                 None,
    "M15":                 None,
}

# Qualifying bonus points (only ATP Tour events have these)
QUALIFYING_POINTS = {
    "Grand Slam (Men's)":  {'Q1': 0, 'Q2': 8,  'Q3': 16, 'qualified': 25},
    "Grand Slam (Women's)":{'Q1': 0, 'Q2': 8,  'Q3': 16, 'qualified': 25},
    "ATP 1000":            {'Q1': 0, 'Q2': 16, 'Q3': 25},
    "WTA 1000":            {'Q1': 0, 'Q2': 16, 'Q3': 25},
    "ATP 500":             {'Q1': 0, 'Q2': 10, 'Q3': 20},
    "WTA 500":             {'Q1': 0, 'Q2': 10, 'Q3': 20},
    "ATP 250":             {'Q1': 0, 'Q2': 6,  'Q3': 12},
    "WTA 250":             {'Q1': 0, 'Q2': 6,  'Q3': 12},
}

# Qualifying field median rank by category (from empirical data)
QUALIFYING_FIELD_RANK = {
    "Grand Slam (Men's)":  178,
    "ATP 1000":            115,
    "ATP 500":             126,
    "ATP 250":             174,
    "Challenger 175":      247,
    "Challenger 125":      410,
    "Challenger 100":      442,
    "Challenger 75":       495,
    "Challenger 50":       633,
}

# Qualifying acceptance: how far below the main draw cutoff can you
# still enter qualifying? Expressed as rank multiplier on main draw cutoff
QUALIFYING_ACCEPTANCE_MULTIPLIER = {
    "Grand Slam (Men's)":  2.0,   # Rank ~500 can enter GS qualifying
    "ATP 1000":            1.8,
    "ATP 500":             1.5,
    "ATP 250":             1.5,
    "Challenger 175":      1.8,
    "Challenger 125":      1.8,
    "Challenger 100":      2.0,
    "Challenger 75":       2.0,
    "Challenger 50":       2.5,
}


def _get_structure(category):
    """Look up qualifying structure for a category."""
    if not isinstance(category, str):
        return None

    if category in QUALIFYING_STRUCTURE:
        return QUALIFYING_STRUCTURE[category]

    # Fuzzy match
    for key, val in QUALIFYING_STRUCTURE.items():
        if key in category:
            return val

    if 'Challenger' in category:
        return QUALIFYING_STRUCTURE.get('Challenger 75')
    return None


def _get_field_rank(category):
    """Get median qualifying field rank."""
    if category in QUALIFYING_FIELD_RANK:
        return QUALIFYING_FIELD_RANK[category]
    for key, rank in QUALIFYING_FIELD_RANK.items():
        if key in (category or ''):
            return rank
    return 500


class QualifyingPathway:
    """
    Models the qualifying draw as an alternative entry pathway.
    """

    def can_enter_qualifying(self, category, player_rank,
                             main_draw_cutoff=None):
        """
        Check if a player can enter the qualifying draw.

        Args:
            category: Tournament category
            player_rank: Player's current ranking
            main_draw_cutoff: Rank cutoff for main draw acceptance
                             (if None, uses category default)

        Returns:
            bool: Whether the player can enter qualifying
        """
        structure = _get_structure(category)
        if structure is None:
            return False  # No qualifying draw (ITF events)

        if main_draw_cutoff is None:
            # Rough defaults
            defaults = {
                "Grand Slam": 250, "ATP 1000": 100, "ATP 500": 100,
                "ATP 250": 200, "Challenger 175": 200, "Challenger 125": 300,
                "Challenger 100": 400, "Challenger 75": 500, "Challenger 50": 800,
            }
            for key, cutoff in defaults.items():
                if key in (category or ''):
                    main_draw_cutoff = cutoff
                    break
            if main_draw_cutoff is None:
                main_draw_cutoff = 500

        mult = QUALIFYING_ACCEPTANCE_MULTIPLIER.get(category, 2.0)
        for key, m in QUALIFYING_ACCEPTANCE_MULTIPLIER.items():
            if key in (category or ''):
                mult = m
                break

        qualifying_cutoff = int(main_draw_cutoff * mult)
        return player_rank <= qualifying_cutoff

    def simulate_qualifying(self, player_rank, category, surface='Hard',
                            rng=None, win_model=None):
        """
        Simulate a qualifying attempt.

        Args:
            player_rank: Player's ranking
            category: Tournament category
            surface: Court surface
            rng: Random number generator
            win_model: WinProbabilityModel instance

        Returns:
            dict with:
                qualified: bool
                rounds_played: int (matches played in qualifying)
                qualifying_points: int (bonus points earned)
                qualifying_prize: int (prize money from qualifying)
        """
        if rng is None:
            rng = random

        structure = _get_structure(category)
        if structure is None:
            return {
                'qualified': False, 'rounds_played': 0,
                'qualifying_points': 0, 'qualifying_prize': 0,
            }

        n_rounds = structure['rounds']
        field_median = _get_field_rank(category)

        # Generate qualifying opponents (weaker than main draw)
        # Qualifying field has higher variance than main draw
        spread = field_median * 0.35

        # Determine tier for win probability model
        if win_model is not None:
            from win_probability import CATEGORY_TO_TIER
            tier = CATEGORY_TO_TIER.get(category, 'Challenger')
        else:
            tier = 'Challenger'

        rounds_played = 0
        qualified = False
        q_points_table = QUALIFYING_POINTS.get(category, {})
        # Fuzzy match
        if not q_points_table:
            for key, table in QUALIFYING_POINTS.items():
                if key in (category or ''):
                    q_points_table = table
                    break

        for r in range(n_rounds):
            rounds_played += 1

            # Opponent from qualifying field (later rounds = slightly stronger)
            strength_shift = -field_median * 0.1 * r  # Gets harder each round
            opp_rank = max(1, int(rng.gauss(field_median + strength_shift, spread)))

            # Calculate win probability
            if win_model is not None:
                p_win = win_model.predict(player_rank, opp_rank, surface, tier)
            else:
                # Simple rank-based estimate
                rank_diff = opp_rank - player_rank
                p_win = 0.5 + rank_diff / (abs(rank_diff) + 200) * 0.3

            if rng.random() >= p_win:
                # Lost in qualifying
                break
        else:
            qualified = True

        # Calculate qualifying points (only for ATP Tour events)
        qualifying_points = 0
        if structure['bonus_points'] and q_points_table:
            if qualified:
                qualifying_points = q_points_table.get('qualified',
                                    q_points_table.get(f'Q{n_rounds}', 0))
            elif rounds_played >= 2:
                qualifying_points = q_points_table.get(f'Q{rounds_played}', 0)

        # Qualifying prize money (typically small at Challengers)
        qualifying_prize = 0
        if 'Challenger' in (category or ''):
            qualifying_prize = rounds_played * 200  # ~$200 per round
        elif 'ATP' in (category or '') or 'Grand Slam' in (category or ''):
            qualifying_prize = rounds_played * 1000  # Higher at ATP level

        return {
            'qualified': qualified,
            'rounds_played': rounds_played,
            'qualifying_points': qualifying_points,
            'qualifying_prize': qualifying_prize,
        }


# =========================================================================
# DEMO
# =========================================================================
if __name__ == '__main__':
    print("Qualifying Pathway - Demo")
    print("=" * 55)

    qp = QualifyingPathway()

    # Test can_enter_qualifying
    print("\nCan enter qualifying?")
    for cat, rank in [("Challenger 75", 400), ("Challenger 75", 800),
                      ("Challenger 75", 1200), ("ATP 250", 200),
                      ("ATP 250", 400), ("M25", 500)]:
        can = qp.can_enter_qualifying(cat, rank)
        print(f"  {cat:<20s} rank {rank:>5d}: {'YES' if can else 'NO'}")

    # Simulate qualifying attempts
    print("\nSimulating 1000 qualifying attempts (Challenger 75, rank 400):")
    successes = 0
    total_rounds = 0
    rng = random.Random(42)
    for _ in range(1000):
        result = qp.simulate_qualifying(400, "Challenger 75", "Clay", rng=rng)
        if result['qualified']:
            successes += 1
        total_rounds += result['rounds_played']
    print(f"  Qualification rate: {successes/10:.1f}%")
    print(f"  Avg rounds played: {total_rounds/1000:.1f}")

    print("\nSimulating 1000 qualifying attempts (ATP 250, rank 180):")
    successes = 0
    total_pts = 0
    rng = random.Random(42)
    for _ in range(1000):
        result = qp.simulate_qualifying(180, "ATP 250", "Hard", rng=rng)
        if result['qualified']:
            successes += 1
        total_pts += result['qualifying_points']
    print(f"  Qualification rate: {successes/10:.1f}%")
    print(f"  Avg qualifying points: {total_pts/1000:.1f}")

    # Rank sensitivity
    print("\nQualification rate by rank (Challenger 75, Clay, 1000 sims):")
    for rank in [200, 300, 400, 500, 600, 800]:
        rng = random.Random(42)
        q = sum(1 for _ in range(1000)
                if qp.simulate_qualifying(rank, "Challenger 75", "Clay", rng=rng)['qualified'])
        print(f"  Rank {rank:>4d}: {q/10:.1f}%")
