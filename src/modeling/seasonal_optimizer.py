"""
Tennis Tournament Optimizer - Seasonal Optimizer
=================================================

The core algorithm. Takes a player's current ranking and a planning window,
generates candidate tournament schedules, simulates each with Monte Carlo,
and recommends the best options.

Architecture: Simulate-and-Score
- Step 1: Build eligible tournament list
- Step 2: Compute per-tournament expected points
- Step 3: Generate N random valid schedules
- Step 4: Simulate each schedule with ranking feedback
- Step 5: Rank and present top schedules

Usage:
    from seasonal_optimizer import SeasonalOptimizer
    
    optimizer = SeasonalOptimizer(calendar_data_path="path/to/atp_clean.csv")
    results = optimizer.optimize(
        player_rank=250,
        player_points=224,
        planning_window=("2025-04-01", "2025-06-15"),
        n_schedules=500,
        n_sims=5000,
    )
"""

import random
import numpy as np
import time
import sys
import os

# Add modeling directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from win_probability import WinProbabilityModel, CATEGORY_TO_TIER
from tournament_economics import get_points_table, get_prize_table
from points_to_rank import PointsRankMapper, ENTRY_THRESHOLDS
from scheduling_constraints import (
    get_scheduling_constraints, get_rank_bracket,
    MIN_REST_DAYS, TOURNAMENTS_PER_YEAR,
    get_surface_weight
)
from travel_costs import TravelCostModel, COUNTRY_CONTINENT


# ==============================================================================
# TOURNAMENT CALENDAR
# ==============================================================================

class TournamentCalendar:
    """
    Manages the tournament calendar for a planning window.
    Extracts from historical match data or accepts manual entries.
    """
    
    def __init__(self):
        self.tournaments = []
    
    def load_from_csv(self, csv_path, year=2025, level='pro'):
        """Load tournament calendar from clean match data."""
        import pandas as pd
        
        cols = ['tournament_name', 'category', 'surface', 'location', 'country',
                'start_date', 'end_date', 'year', 'tier', 'tier_name', 'level',
                'mandatory', 'round', 'player_rank']
        
        df = pd.read_csv(csv_path, usecols=cols, low_memory=False)
        df = df[(df['year'] == year) & (df['level'] == level)]
        
        # One row per tournament
        cal = df.groupby(['tournament_name', 'category']).agg(
            surface=('surface', 'first'),
            location=('location', 'first'),
            country=('country', 'first'),
            start_date=('start_date', 'first'),
            end_date=('end_date', 'first'),
            tier=('tier', 'first'),
            tier_name=('tier_name', 'first'),
            mandatory=('mandatory', 'first'),
            n_matches=('round', 'count'),
            rounds=('round', lambda x: sorted(x.unique())),
            median_field_rank=('player_rank', 'median'),
            field_p25=('player_rank', lambda x: x.quantile(0.25)),
            field_p75=('player_rank', lambda x: x.quantile(0.75)),
        ).reset_index()
        
        # Parse dates
        cal['start_dt'] = pd.to_datetime(
            cal['start_date'] + ' ' + str(year), format='%b %d %Y', errors='coerce')
        cal['end_dt'] = pd.to_datetime(
            cal['end_date'] + ' ' + str(year), format='%b %d %Y', errors='coerce')
        
        # Fix year boundary
        wrap = cal['end_dt'].notna() & cal['start_dt'].notna() & (cal['end_dt'] < cal['start_dt'])
        cal.loc[wrap, 'end_dt'] = pd.to_datetime(
            cal.loc[wrap, 'end_date'] + ' ' + str(year + 1), format='%b %d %Y', errors='coerce')
        
        # Week number
        cal['week'] = cal['start_dt'].dt.isocalendar().week.astype(int)
        
        # Estimate draw size
        def est_draw(rounds):
            if '1/64' in rounds: return 128
            elif '1/32' in rounds: return 64
            elif '1/16' in rounds: return 32
            else: return 16
        
        cal['draw_size'] = cal['rounds'].apply(est_draw)
        
        self.tournaments = cal.to_dict('records')
        return self
    
    def load_synthetic(self, tournaments_list):
        """
        Load tournaments from a list of dicts.
        Each dict needs: tournament_name, category, surface, week,
                         median_field_rank, draw_size, tier_name, mandatory
        """
        self.tournaments = tournaments_list
        return self
    
    def get_eligible(self, player_rank, planning_start_week, planning_end_week,
                     surface_filter=None, exclude_tournaments=None):
        """Filter tournaments to those eligible and in the planning window."""
        eligible = []
        mapper = PointsRankMapper()
        
        for t in self.tournaments:
            week = t.get('week', 0)
            if week < planning_start_week or week > planning_end_week:
                continue
            
            # Ranking eligibility - use actual category for precise thresholds
            category = t.get('category', '')
            
            if not mapper.can_enter(player_rank, category):
                continue
            
            # Surface filter
            if surface_filter and t.get('surface', '') not in surface_filter:
                continue
            
            # Exclusions
            if exclude_tournaments and t.get('tournament_name', '') in exclude_tournaments:
                continue
            
            eligible.append(t)
        
        return eligible
    
    def group_by_week(self, tournaments):
        """Group tournaments by calendar week."""
        by_week = {}
        for t in tournaments:
            week = t.get('week', 0)
            if week not in by_week:
                by_week[week] = []
            by_week[week].append(t)
        return by_week


# ==============================================================================
# TOURNAMENT SIMULATOR
# ==============================================================================

class TournamentSimulator:
    """
    Simulates a player's progression through a single tournament bracket.
    Uses tournament-specific field strength when available.
    """
    
    def __init__(self, win_model=None, field_data_path=None, category_fallback_path=None,
                 name_to_key_path=None):
        self.win_model = win_model or WinProbabilityModel()
        self.field_profiles = {}
        self.category_fallbacks = {}
        self.name_to_key = {}
        
        # Load field profiles (keyed by "location|tier_group")
        if field_data_path and os.path.exists(field_data_path):
            import json
            with open(field_data_path) as f:
                self.field_profiles = json.load(f)
        
        if category_fallback_path and os.path.exists(category_fallback_path):
            import json
            with open(category_fallback_path) as f:
                self.category_fallbacks = json.load(f)
        
        # Load tournament name -> field key mapping
        if name_to_key_path and os.path.exists(name_to_key_path):
            import json
            with open(name_to_key_path) as f:
                self.name_to_key = json.load(f)
    
    def _get_field_profile(self, tournament):
        """
        Get field strength profile for a tournament.
        Hierarchy:
          1. Name -> key mapping -> field profile (exact match)
          2. Category fallback with rank adjustment
          3. Default values from tournament dict
        Uses a cache for performance.
        """
        name = tournament.get('tournament_name', '')
        category = tournament.get('category', '')
        cache_key = f"{name}|{category}"
        
        if not hasattr(self, '_field_cache'):
            self._field_cache = {}
        if cache_key in self._field_cache:
            return self._field_cache[cache_key]
        
        # 1. Try name-to-key mapping -> field profile
        if name in self.name_to_key:
            field_key = self.name_to_key[name]
            if field_key in self.field_profiles:
                self._field_cache[cache_key] = self.field_profiles[field_key]
                return self._field_cache[cache_key]
        
        # 2. Category fallback
        TIER_GROUPS = {
            'Grand Slam': 'Grand Slam', 'Masters 1000': 'Masters 1000',
            'ATP 500': 'ATP 500', 'ATP 250': 'ATP 250',
            'Challenger': 'Challenger', 'ITF': 'ITF',
        }
        tier_name = tournament.get('tier_name', '')
        tier_group = CATEGORY_TO_TIER.get(category, tier_name)
        
        if tier_group in self.category_fallbacks:
            result = self.category_fallbacks[tier_group].copy()
            self._field_cache[cache_key] = result
            return result
        
        # 3. Fall back to tournament dict values
        result = {
            'median_rank': tournament.get('median_field_rank', 300),
            'p25_rank': tournament.get('field_p25', 150),
            'p75_rank': tournament.get('field_p75', 500),
            'p10_rank': tournament.get('field_p25', 150) * 0.6,
            'p90_rank': tournament.get('field_p75', 500) * 1.3,
        }
        self._field_cache[cache_key] = result
        return result
    
    def _generate_field(self, tournament, draw_size, rng_seed=42):
        """Pre-generate a field of opponent ranks for a tournament."""
        cache_key = tournament.get('tournament_name', '') + '|' + str(draw_size)
        if not hasattr(self, '_draw_cache'):
            self._draw_cache = {}
        if cache_key in self._draw_cache:
            return self._draw_cache[cache_key]
        
        field_profile = self._get_field_profile(tournament)
        median_field = field_profile.get('median_rank', 300)
        field_p25 = field_profile.get('p25_rank', median_field * 0.5)
        field_p75 = field_profile.get('p75_rank', median_field * 1.5)
        field_p10 = field_profile.get('p10_rank', field_p25 * 0.6)
        field_p90 = field_profile.get('p90_rank', field_p75 * 1.3)
        
        rng = random.Random(rng_seed)
        n_seeds = min(8, draw_size // 4)
        
        field = []
        for i in range(draw_size):
            if i < n_seeds:
                rank = max(1, int(rng.gauss(
                    field_p10 + (field_p25 - field_p10) * i / max(n_seeds - 1, 1),
                    (field_p25 - field_p10) * 0.3)))
            else:
                rank = max(1, int(rng.gauss(
                    (median_field + field_p75) / 2,
                    (field_p90 - median_field) * 0.4)))
            field.append(rank)
        
        field.sort()
        self._draw_cache[cache_key] = field
        return field

    def simulate_once(self, player_rank, tournament, rng=None):
        """
        Simulate one run through a tournament bracket with realistic seeded draw.
        
        Returns:
            dict with round_reached, points_earned, prize_earned
        """
        if rng is None:
            rng = random
        
        category = tournament.get('category', '')
        surface = tournament.get('surface', 'Hard')
        tier_group = CATEGORY_TO_TIER.get(category, tournament.get('tier_name', 'Challenger'))
        draw_size = tournament.get('draw_size', 32)
        
        # Use pre-generated field
        field = self._generate_field(tournament, draw_size)
        
        # Determine round structure
        if draw_size >= 128:
            rounds = ['1/64', '1/32', '1/16', '1/8', 'QF', 'SF', 'F', 'W']
        elif draw_size >= 64:
            rounds = ['1/32', '1/16', '1/8', 'QF', 'SF', 'F', 'W']
        elif draw_size >= 32:
            rounds = ['1/16', '1/8', 'QF', 'SF', 'F', 'W']
        else:
            rounds = ['1/8', 'QF', 'SF', 'F', 'W']
        
        n_seeds = min(8, draw_size // 4)
        n_rounds = len(rounds) - 1  # exclude 'W' which is winning the final
        
        points_table = get_points_table(category) or {}
        prize_table = get_prize_table(category) or {}
        
        # Simulate match-by-match with bracket logic
        # R1: unseeded player likely faces a seed (~25% chance against top seed)
        # Later rounds: opponents get progressively stronger (survivors)
        round_reached = rounds[0]
        
        for r_idx in range(n_rounds):
            round_reached = rounds[r_idx]
            
            # Pick opponent based on round
            # R1: random from the full unseeded portion, or a seed
            # Later rounds: pick from the stronger portion (survivors)
            if r_idx == 0:
                # R1: 25% chance of facing a top-8 seed, otherwise random unseeded
                if rng.random() < 0.25 and n_seeds > 0:
                    opp_rank = field[rng.randint(0, n_seeds - 1)]
                else:
                    opp_rank = field[rng.randint(n_seeds, len(field) - 1)]
            else:
                # Later rounds: opponent is a "survivor" - pick from top portion
                # Each round halves the field, survivors are on average stronger
                survivor_pool_size = max(1, len(field) // (2 ** r_idx))
                opp_idx = rng.randint(0, survivor_pool_size - 1)
                opp_rank = field[opp_idx]
            
            p_win = self.win_model.predict(player_rank, opp_rank, surface, tier_group)
            
            if rng.random() >= p_win:
                break
        else:
            # Won all rounds including the final
            round_reached = rounds[-1]  # 'W'
        
        # Calculate rewards
        points_earned = points_table.get(round_reached, 0)
        prize_earned = prize_table.get(round_reached, 0)
        
        return {
            'round_reached': round_reached,
            'points_earned': points_earned,
            'prize_earned': prize_earned,
        }
    
    def estimate_ev(self, player_rank, tournament, n_sims=1000, seed=None):
        """
        Estimate expected value for a single tournament.
        
        Returns:
            dict with expected_points, expected_prize, round_probabilities
        """
        rng = random.Random(seed)
        
        total_points = 0.0
        total_prize = 0.0
        round_counts = {}
        
        for _ in range(n_sims):
            result = self.simulate_once(player_rank, tournament, rng)
            total_points += result['points_earned']
            total_prize += result['prize_earned']
            
            r = result['round_reached']
            round_counts[r] = round_counts.get(r, 0) + 1
        
        round_probs = {r: c / n_sims for r, c in sorted(round_counts.items())}
        
        return {
            'expected_points': round(total_points / n_sims, 2),
            'expected_prize': round(total_prize / n_sims, 2),
            'round_probs': round_probs,
            'n_sims': n_sims,
        }


# ==============================================================================
# SCHEDULE GENERATOR
# ==============================================================================

class ScheduleGenerator:
    """
    Generates random valid tournament schedules with geographic coherence.
    """
    
    def __init__(self, tournaments_by_week, mandatory_weeks=None,
                 travel_model=None, player_rank=500):
        self.by_week = tournaments_by_week
        self.mandatory_weeks = mandatory_weeks or {}  # {week: tournament}
        self.travel_model = travel_model
        self.player_rank = player_rank
        self.all_weeks = sorted(set(
            list(self.by_week.keys()) + list(self.mandatory_weeks.keys())
        ))
    
    def _get_continent(self, tournament):
        """Get continent for a tournament."""
        country = tournament.get('country', '')
        if not isinstance(country, str):
            country = ''
        return COUNTRY_CONTINENT.get(country, 'Unknown')
    
    def generate(self, tournament_evs, target_tournaments=8,
                 max_consecutive=4, max_continent_switches=1, rng=None):
        """
        Generate one random valid schedule with geographic coherence.
        
        Args:
            tournament_evs: dict mapping tournament_name -> expected_points
            target_tournaments: desired number of tournaments
            max_consecutive: max back-to-back weeks
            max_continent_switches: max allowed continent changes in schedule
            rng: random number generator
        
        Returns:
            list of (week, tournament_dict) tuples
        """
        if rng is None:
            rng = random
        
        schedule = []
        consecutive = 0
        tournaments_so_far = 0
        weeks_remaining = len(self.all_weeks)
        current_continent = None
        continent_switches = 0
        
        for week in self.all_weeks:
            weeks_remaining -= 1
            
            # Mandatory event
            if week in self.mandatory_weeks:
                t = self.mandatory_weeks[week]
                new_continent = self._get_continent(t)
                if current_continent and new_continent != current_continent:
                    continent_switches += 1
                current_continent = new_continent
                schedule.append((week, t))
                consecutive += 1
                tournaments_so_far += 1
                continue
            
            # Force rest if at max consecutive
            if consecutive >= max_consecutive:
                consecutive = 0
                continue
            
            # No tournaments available this week
            if week not in self.by_week or len(self.by_week[week]) == 0:
                consecutive = 0
                continue
            
            # Decide: play or rest?
            tournaments_needed = target_tournaments - tournaments_so_far
            if tournaments_needed <= 0:
                consecutive = 0
                continue
            
            if weeks_remaining <= 0:
                play_prob = 1.0
            else:
                # Higher probability of playing if we're behind target
                play_prob = min(0.95, tournaments_needed / (weeks_remaining + 1) + 0.2)
            
            if rng.random() > play_prob:
                consecutive = 0
                continue
            
            # Pick a tournament (weighted by EV + geography + surface season)
            options = self.by_week[week]
            weights = []
            for t in options:
                name = t.get('tournament_name', '')
                ev = tournament_evs.get(name, 1.0)
                base_weight = max(0.1, ev)
                
                # Surface seasonal weighting
                t_surface = t.get('surface', 'Hard')
                surf_mult = get_surface_weight(t_surface, week, self.player_rank)
                
                # Geographic weighting
                if current_continent is not None:
                    t_continent = self._get_continent(t)
                    
                    if t_continent == current_continent:
                        # Same continent: bonus for same country
                        t_country = t.get('country', '')
                        if not isinstance(t_country, str):
                            t_country = ''
                        if self.travel_model and t_country == self.travel_model.player_country:
                            geo_mult = 3.0  # Strong preference for home country
                        else:
                            geo_mult = 2.0  # Strong preference for same continent
                    else:
                        # Different continent: check if we can still switch
                        if continent_switches >= max_continent_switches:
                            geo_mult = 0.01  # Nearly block it
                        else:
                            geo_mult = 0.15  # Heavy penalty
                else:
                    # First tournament: strongly prefer player's home continent
                    if self.travel_model:
                        t_continent = self._get_continent(t)
                        if t_continent == self.travel_model.player_continent:
                            geo_mult = 3.0
                        else:
                            geo_mult = 0.1  # Rarely start on a different continent
                    else:
                        geo_mult = 1.0
                
                weights.append(base_weight * geo_mult * surf_mult)
            
            # Weighted random selection
            total_w = sum(weights)
            if total_w <= 0:
                consecutive = 0
                continue
            
            r = rng.random() * total_w
            cumulative = 0
            chosen = options[0]
            for t, w in zip(options, weights):
                cumulative += w
                if r <= cumulative:
                    chosen = t
                    break
            
            # Track continent switches
            new_continent = self._get_continent(chosen)
            if current_continent and new_continent != current_continent:
                continent_switches += 1
            current_continent = new_continent
            
            schedule.append((week, chosen))
            consecutive += 1
            tournaments_so_far += 1
        
        return schedule


# ==============================================================================
# SEASONAL OPTIMIZER
# ==============================================================================

class SeasonalOptimizer:
    """
    The main optimizer. Generates and evaluates tournament schedules.
    """
    
    def __init__(self, field_data_path=None, category_fallback_path=None,
                 name_to_key_path=None, player_country='FRA'):
        self.win_model = WinProbabilityModel()
        self.player_country = player_country
        self.travel_model = TravelCostModel(player_country=player_country)
        
        # Default paths relative to this file's location
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                  '..', '..', 'models')
        if field_data_path is None:
            field_data_path = os.path.join(model_dir, 'field_profiles.json')
        if category_fallback_path is None:
            category_fallback_path = os.path.join(model_dir, 'category_field_fallbacks.json')
        if name_to_key_path is None:
            name_to_key_path = os.path.join(model_dir, 'tournament_name_to_key.json')
        
        self.simulator = TournamentSimulator(
            self.win_model, field_data_path, category_fallback_path, name_to_key_path)
        self.mapper = PointsRankMapper()
        self.calendar = TournamentCalendar()
    
    def load_calendar(self, csv_path, year=2025):
        """Load tournament calendar from clean match data."""
        self.calendar.load_from_csv(csv_path, year=year)
        return self
    
    def load_synthetic_calendar(self, tournaments):
        """Load a synthetic tournament calendar."""
        self.calendar.load_synthetic(tournaments)
        return self
    
    def optimize(self, player_rank, player_points,
                 planning_start_week, planning_end_week,
                 n_schedules=500, n_sims_per_tournament=1000,
                 n_sims_per_schedule=5000, target_tournaments=None,
                 surface_filter=None, exclude_tournaments=None,
                 max_continent_switches=1,
                 seed=None, verbose=True):
        """
        Run the full optimization pipeline.
        
        Args:
            player_rank: Current ATP ranking
            player_points: Current ranking points total
            planning_start_week: Start week of planning window
            planning_end_week: End week of planning window
            n_schedules: Number of candidate schedules to generate
            n_sims_per_tournament: Monte Carlo sims for per-tournament EV
            n_sims_per_schedule: Monte Carlo sims for full schedule evaluation
            target_tournaments: Target number of tournaments (auto if None)
            surface_filter: List of surfaces to include (None = all)
            exclude_tournaments: Tournament names to exclude
            max_continent_switches: Max continent changes per schedule (default 1)
            seed: Random seed for reproducibility
            verbose: Print progress
        
        Returns:
            dict with top schedules, per-tournament EVs, and metadata
        """
        t_total = time.time()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Determine target tournament count
        if target_tournaments is None:
            bracket = get_rank_bracket(player_rank)
            target_tournaments = TOURNAMENTS_PER_YEAR[bracket]["median"]
            # Scale to window size
            window_weeks = planning_end_week - planning_start_week + 1
            target_tournaments = max(3, min(
                int(target_tournaments * window_weeks / 44), window_weeks - 2))
        
        if verbose:
            print(f"Tennis Tournament Optimizer")
            print(f"{'='*60}")
            print(f"  Player: rank {player_rank}, {player_points} points")
            print(f"  Home country: {self.player_country} ({self.travel_model.player_continent})")
            print(f"  Window: weeks {planning_start_week}-{planning_end_week} "
                  f"({planning_end_week - planning_start_week + 1} weeks)")
            print(f"  Target: {target_tournaments} tournaments")
            print(f"  Max continent switches: {max_continent_switches}")
            print(f"  Schedules to generate: {n_schedules}")
            print(f"  Sims per schedule: {n_sims_per_schedule}")
        
        # =====================================================================
        # STEP 1: Build eligible tournament list
        # =====================================================================
        if verbose:
            print(f"\n[Step 1] Filtering eligible tournaments...")
        
        eligible = self.calendar.get_eligible(
            player_rank, planning_start_week, planning_end_week,
            surface_filter, exclude_tournaments)
        
        by_week = self.calendar.group_by_week(eligible)
        
        # Find mandatory events
        mandatory_weeks = {}
        for t in eligible:
            if t.get('mandatory') in ('ranking', 'commitment_top30'):
                mandatory_weeks[t['week']] = t
        
        if verbose:
            print(f"  {len(eligible)} eligible tournaments across "
                  f"{len(by_week)} weeks")
            if mandatory_weeks:
                print(f"  {len(mandatory_weeks)} mandatory events locked in")
        
        if len(eligible) == 0:
            if verbose:
                print("  No eligible tournaments found!")
            return {"error": "No eligible tournaments in window"}
        
        # =====================================================================
        # STEP 2: Compute per-tournament EV (adjusted for acceptance probability)
        # =====================================================================
        if verbose:
            print(f"\n[Step 2] Computing per-tournament EV "
                  f"({n_sims_per_tournament} sims each)...")
        
        t0 = time.time()
        tournament_evs = {}       # Effective EV (raw × acceptance prob)
        tournament_raw_evs = {}   # Raw EV (if accepted)
        tournament_accept = {}    # Acceptance probability
        tournament_details = {}
        
        for t in eligible:
            name = t.get('tournament_name', '')
            if name in tournament_evs:
                continue  # Already computed
            
            ev = self.simulator.estimate_ev(
                player_rank, t, n_sims=n_sims_per_tournament)
            
            category = t.get('category', '')
            accept_prob = self.mapper.acceptance_probability(player_rank, category)
            
            tournament_raw_evs[name] = ev['expected_points']
            tournament_accept[name] = accept_prob
            tournament_evs[name] = ev['expected_points'] * accept_prob
            tournament_details[name] = ev
            tournament_details[name]['acceptance_probability'] = accept_prob
        
        if verbose:
            print(f"  {len(tournament_evs)} tournaments evaluated "
                  f"({time.time()-t0:.1f}s)")
            
            # Show top 10 by effective EV
            sorted_evs = sorted(tournament_evs.items(), key=lambda x: -x[1])
            print(f"\n  Top 10 by effective EV (raw × acceptance):")
            for name, eff_ev in sorted_evs[:10]:
                t_info = next(t for t in eligible if t.get('tournament_name') == name)
                raw = tournament_raw_evs[name]
                acc = tournament_accept[name]
                print(f"    {name:<35s} EV={eff_ev:>5.1f} "
                      f"(raw={raw:.1f} × {acc:.0%} accept)  "
                      f"{t_info.get('category','?')}")
        
        # =====================================================================
        # STEP 3: Generate candidate schedules
        # =====================================================================
        if verbose:
            print(f"\n[Step 3] Generating {n_schedules} candidate schedules...")
        
        t0 = time.time()
        generator = ScheduleGenerator(by_week, mandatory_weeks,
                                      travel_model=self.travel_model,
                                      player_rank=player_rank)
        
        candidates = []
        for i in range(n_schedules):
            rng = random.Random(seed + i if seed else None)
            schedule = generator.generate(
                tournament_evs, target_tournaments=target_tournaments,
                max_continent_switches=max_continent_switches, rng=rng)
            if len(schedule) >= 2:  # At least 2 tournaments
                candidates.append(schedule)
        
        if verbose:
            sizes = [len(s) for s in candidates]
            print(f"  {len(candidates)} valid schedules generated "
                  f"({time.time()-t0:.1f}s)")
            print(f"  Tournament count: min={min(sizes)}, "
                  f"median={sorted(sizes)[len(sizes)//2]}, max={max(sizes)}")
        
        # =====================================================================
        # STEP 4: Simulate each schedule with ranking feedback
        # =====================================================================
        if verbose:
            print(f"\n[Step 4] Simulating {len(candidates)} schedules "
                  f"({n_sims_per_schedule} sims each)...")
        
        t0 = time.time()
        schedule_results = []
        
        for sched_idx, schedule in enumerate(candidates):
            if verbose and (sched_idx + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (sched_idx + 1) / elapsed
                remaining = (len(candidates) - sched_idx - 1) / rate
                print(f"    {sched_idx+1}/{len(candidates)} "
                      f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")
            
            sim_points = []
            sim_prizes = []
            sim_final_ranks = []
            sim_round_results = []
            
            for sim in range(n_sims_per_schedule):
                rng = random.Random(
                    (seed or 0) + sched_idx * 100000 + sim)
                
                current_rank = player_rank
                current_points = player_points
                total_points_gained = 0
                total_prize = 0
                round_results = []
                
                # Estimate weekly points expiry: assume points are spread
                # roughly evenly across 44 active weeks per year
                weekly_expiry = player_points / 44.0 if player_points > 0 else 0
                
                last_week = None
                for week, tournament in schedule:
                    # Estimate points dropping off since last tournament
                    if last_week is not None:
                        weeks_elapsed = week - last_week
                        points_expiring = weekly_expiry * weeks_elapsed
                    else:
                        points_expiring = 0
                    last_week = week
                    
                    # Check acceptance: randomly determine if player gets in
                    t_name = tournament.get('tournament_name', '')
                    accept_prob = tournament_accept.get(t_name, 1.0)
                    if accept_prob < 1.0 and rng.random() > accept_prob:
                        # Not accepted — points still expire, but no play
                        current_points = current_points - points_expiring
                        current_points = max(0, current_points)
                        current_rank = self.mapper.points_to_rank(current_points)
                        continue
                    
                    result = self.simulator.simulate_once(
                        current_rank, tournament, rng)
                    
                    total_points_gained += result['points_earned']
                    total_prize += result['prize_earned']
                    current_points = current_points + result['points_earned'] - points_expiring
                    current_points = max(0, current_points)
                    current_rank = self.mapper.points_to_rank(current_points)
                    
                    round_results.append(result['round_reached'])
                
                sim_points.append(total_points_gained)
                sim_prizes.append(total_prize)
                sim_final_ranks.append(current_rank)
            
            schedule_results.append({
                'schedule': schedule,
                'tournaments': [t.get('tournament_name', '?') for _, t in schedule],
                'n_tournaments': len(schedule),
                'expected_points': float(np.mean(sim_points)),
                'points_p20': float(np.percentile(sim_points, 20)),
                'points_p50': float(np.median(sim_points)),
                'points_p80': float(np.percentile(sim_points, 80)),
                'expected_prize': float(np.mean(sim_prizes)),
                'expected_final_rank': float(np.mean(sim_final_ranks)),
                'final_rank_p20': float(np.percentile(sim_final_ranks, 20)),
                'final_rank_p80': float(np.percentile(sim_final_ranks, 80)),
                'travel_info': self.travel_model.get_schedule_travel_info(schedule),
            })
            # Compute net ROI
            sr = schedule_results[-1]
            sr['travel_cost'] = sr['travel_info']['total_cost']
            sr['net_prize'] = sr['expected_prize'] - sr['travel_cost']
            # Combined score: points are primary, net_prize breaks ties
            sr['combined_score'] = sr['expected_points'] + sr['net_prize'] / 500.0
        
        if verbose:
            print(f"  Done ({time.time()-t0:.1f}s)")
        
        # =====================================================================
        # STEP 5: Rank and select top schedules
        # =====================================================================
        if verbose:
            print(f"\n[Step 5] Ranking schedules...")
        
        # Sort by combined score (points + net ROI)
        schedule_results.sort(key=lambda x: -x['combined_score'])
        
        # Select top diverse schedules
        top_schedules = self._select_diverse_top(schedule_results, n_top=5)
        
        total_time = time.time() - t_total
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"RESULTS (total time: {total_time:.0f}s)")
            print(f"{'='*60}")
            
            for i, sched in enumerate(top_schedules):
                print(f"\n--- Schedule {i+1} ---")
                print(f"  Expected points: {sched['expected_points']:.1f} "
                      f"(range: {sched['points_p20']:.0f} - {sched['points_p80']:.0f})")
                print(f"  Expected prize:  ${sched['expected_prize']:,.0f}")
                print(f"  Travel cost:     ${sched['travel_cost']:,.0f}  "
                      f"({sched['travel_info']['tier_breakdown']})")
                print(f"  Net prize (ROI): ${sched['net_prize']:,.0f}")
                print(f"  Expected rank:   {sched['expected_final_rank']:.0f} "
                      f"(range: {sched['final_rank_p80']:.0f} - {sched['final_rank_p20']:.0f})")
                print(f"  Tournaments ({sched['n_tournaments']}):")
                
                for week, tournament in sched['schedule']:
                    name = tournament.get('tournament_name', '?')
                    cat = tournament.get('category', '?')
                    surf = tournament.get('surface', '?')
                    country = tournament.get('country', '?')
                    ev = tournament_evs.get(name, 0)
                    acc = tournament_accept.get(name, 1.0)
                    acc_str = f'{acc:.0%}' if acc < 1.0 else '100%'
                    print(f"    Week {week:>2d}: {name:<35s} "
                          f"{cat:<15s} {surf:<8s} {country:<4s} EV={ev:.1f} ({acc_str})")
        
        return {
            'top_schedules': top_schedules,
            'all_results': schedule_results[:20],  # Top 20 for reference
            'tournament_evs': tournament_evs,
            'tournament_raw_evs': tournament_raw_evs,
            'tournament_accept': tournament_accept,
            'tournament_details': tournament_details,
            'metadata': {
                'player_rank': player_rank,
                'player_points': player_points,
                'player_country': self.player_country,
                'planning_window': (planning_start_week, planning_end_week),
                'max_continent_switches': max_continent_switches,
                'n_eligible': len(eligible),
                'n_schedules_generated': len(candidates),
                'n_sims_per_schedule': n_sims_per_schedule,
                'total_time_seconds': round(total_time, 1),
            }
        }
    
    def _select_diverse_top(self, sorted_results, n_top=5):
        """Select top N schedules that are meaningfully different."""
        if len(sorted_results) <= n_top:
            return sorted_results
        
        selected = [sorted_results[0]]  # Always include the best
        
        for candidate in sorted_results[1:]:
            if len(selected) >= n_top:
                break
            
            # Check if this schedule is different enough from already selected
            is_diverse = True
            cand_set = set(candidate['tournaments'])
            
            for existing in selected:
                existing_set = set(existing['tournaments'])
                overlap = len(cand_set & existing_set) / max(len(cand_set), 1)
                if overlap > 0.7:  # More than 70% overlap = too similar
                    is_diverse = False
                    break
            
            if is_diverse:
                selected.append(candidate)
        
        # If we still need more, fill with top remaining regardless of diversity
        if len(selected) < n_top:
            for candidate in sorted_results:
                if candidate not in selected:
                    selected.append(candidate)
                if len(selected) >= n_top:
                    break
        
        return selected


# ==============================================================================
# DEMO
# ==============================================================================
if __name__ == '__main__':
    print("Tennis Tournament Optimizer - Demo")
    print("=" * 60)
    
    # Create a synthetic calendar (since we might not have clean CSV here)
    # Simulates a clay season: weeks 14-24
    tournaments = []
    
    # Real-ish clay season tournaments
    clay_season = [
        # Week, Name, Category, Surface, Median field rank, Country
        (14, "Challenger Marbella", "Challenger 80", "Clay", 280, "ESP"),
        (14, "M25 Antalya", "M25", "Clay", 450, "TUR"),
        (15, "Challenger Bordeaux", "Challenger 100", "Clay", 250, "FRA"),
        (15, "M25 Monastir I", "M25", "Clay", 500, "TUN"),
        (16, "Challenger Aix-en-Provence", "Challenger 125", "Clay", 220, "FRA"),
        (16, "M25 Santa Margherita", "M25", "Clay", 480, "ITA"),
        (17, "Barcelona Open", "ATP 500", "Clay", 60, "ESP"),
        (17, "Challenger Francavilla", "Challenger 80", "Clay", 300, "ITA"),
        (17, "M25 Monastir II", "M25", "Clay", 500, "TUN"),
        (18, "Madrid Open", "ATP 1000", "Clay", 40, "ESP"),
        (18, "Challenger Ostrava", "Challenger 100", "Clay", 270, "CZE"),
        (19, "Challenger Lyon", "Challenger 100", "Clay", 240, "FRA"),
        (19, "M25 Tunis", "M25", "Clay", 450, "TUN"),
        (20, "Rome Masters", "ATP 1000", "Clay", 35, "ITA"),
        (20, "Challenger Prague", "Challenger 80", "Clay", 290, "CZE"),
        (21, "Challenger Geneva", "Challenger 100", "Clay", 250, "SUI"),
        (21, "M25 Koper", "M25", "Clay", 470, "SLO"),
        (22, "Roland Garros Qualifying", "Grand Slam (Men's)", "Clay", 150, "FRA"),
        (22, "Challenger Heilbronn", "Challenger 80", "Clay", 300, "GER"),
        (23, "Challenger Parma", "Challenger 100", "Clay", 260, "ITA"),
        (23, "M25 Madrid", "M25", "Clay", 500, "ESP"),
        (24, "Challenger Prostejov", "Challenger 80", "Clay", 290, "CZE"),
        (24, "M25 Hammamet", "M25", "Clay", 480, "TUN"),
    ]
    
    for week, name, category, surface, median_rank, country in clay_season:
        tier_group = CATEGORY_TO_TIER.get(category, 'Challenger')
        tournaments.append({
            'tournament_name': name,
            'category': category,
            'surface': surface,
            'tier_name': tier_group,
            'week': week,
            'median_field_rank': median_rank,
            'field_p25': median_rank * 0.5,
            'field_p75': median_rank * 1.5,
            'draw_size': 128 if 'Grand Slam' in category else 32,
            'mandatory': 'ranking' if 'Grand Slam' in category else 'optional',
            'location': name.split()[-1],
            'country': country,
        })
    
    # Run optimizer
    optimizer = SeasonalOptimizer(player_country='FRA')
    optimizer.load_synthetic_calendar(tournaments)
    
    results = optimizer.optimize(
        player_rank=250,
        player_points=224,
        planning_start_week=14,
        planning_end_week=24,
        n_schedules=200,
        n_sims_per_tournament=500,
        n_sims_per_schedule=2000,
        target_tournaments=7,
        seed=42,
        verbose=True,
    )
    
    print(f"\n\nOptimization complete!")
    print(f"Total time: {results['metadata']['total_time_seconds']:.1f}s")
