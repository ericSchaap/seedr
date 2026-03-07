"""
Tennis Tournament Optimizer - Win Probability Engine
====================================================

Core function: win_probability(player_rank, opponent_rank, surface, tier)

Model: Logistic regression on log(opponent_rank / player_rank),
fitted separately per tier×surface combination.

Hierarchy:
  1. Tier×Surface model (if available and n >= 500)
  2. Tier-only model
  3. Global model

Trained on 1,014,096 ATP pro-vs-pro completed matches (2007-2026).

Usage:
    from win_probability import WinProbabilityModel
    model = WinProbabilityModel()
    p = model.predict(player_rank=150, opponent_rank=45, surface='Clay', tier='Challenger')
"""

import json
import math
import os

# Model parameters (embedded for portability)
# Format: {key: {a: intercept, b: slope, n: training_size, brier: brier_score}}
MODEL_PARAMS = {
    "global": {"a": -0.000854, "b": 1.119926, "n": 1014096, "brier": 0.2139},
    "tier_Grand Slam": {"a": -0.002200, "b": 0.820400, "n": 34417, "brier": 0.2056},
    "tier_Masters 1000": {"a": -0.001700, "b": 0.584400, "n": 27927, "brier": 0.2180},
    "tier_ATP 500": {"a": -0.003800, "b": 0.656600, "n": 19549, "brier": 0.2112},
    "tier_ATP 250": {"a": -0.000500, "b": 0.677800, "n": 64705, "brier": 0.2200},
    "tier_Challenger": {"a": -0.001700, "b": 1.022900, "n": 268173, "brier": 0.2207},
    "tier_ITF": {"a": -0.000200, "b": 1.464800, "n": 596778, "brier": 0.2078},
    "Grand Slam|Clay": {"a": 0.000300, "b": 0.837900, "n": 8710, "brier": 0.2053},
    "Grand Slam|Hard": {"a": -0.001200, "b": 0.837400, "n": 17456, "brier": 0.2043},
    "Grand Slam|Grass": {"a": -0.007000, "b": 0.764400, "n": 8211, "brier": 0.2090},
    "Masters 1000|Clay": {"a": 0.000100, "b": 0.627000, "n": 8870, "brier": 0.2145},
    "Masters 1000|Hard": {"a": -0.003200, "b": 0.554700, "n": 16579, "brier": 0.2199},
    "Masters 1000|Hard (I)": {"a": 0.002300, "b": 0.650500, "n": 2478, "brier": 0.2170},
    "ATP 500|Clay": {"a": 0.001800, "b": 0.695600, "n": 5692, "brier": 0.2130},
    "ATP 500|Hard": {"a": -0.010100, "b": 0.705200, "n": 7002, "brier": 0.2041},
    "ATP 500|Hard (I)": {"a": -0.002700, "b": 0.621100, "n": 5204, "brier": 0.2139},
    "ATP 500|Grass": {"a": -0.001400, "b": 0.478300, "n": 1651, "brier": 0.2242},
    "ATP 250|Clay": {"a": 0.002000, "b": 0.693700, "n": 22313, "brier": 0.2209},
    "ATP 250|Hard": {"a": -0.002400, "b": 0.659300, "n": 17815, "brier": 0.2203},
    "ATP 250|Hard (I)": {"a": -0.003500, "b": 0.755500, "n": 15838, "brier": 0.2139},
    "ATP 250|Grass": {"a": 0.002300, "b": 0.541200, "n": 7896, "brier": 0.2274},
    "Challenger|Clay": {"a": -0.001100, "b": 1.037600, "n": 126215, "brier": 0.2202},
    "Challenger|Hard": {"a": -0.002300, "b": 1.068700, "n": 92677, "brier": 0.2182},
    "Challenger|Hard (I)": {"a": -0.002900, "b": 0.878900, "n": 40134, "brier": 0.2276},
    "Challenger|Grass": {"a": 0.001600, "b": 0.799700, "n": 4467, "brier": 0.2295},
    "Challenger|Carpet": {"a": -0.001800, "b": 1.250200, "n": 3042, "brier": 0.2095},
    "ITF|Clay": {"a": -0.000500, "b": 1.509100, "n": 299644, "brier": 0.2059},
    "ITF|Hard": {"a": 0.000500, "b": 1.467900, "n": 229234, "brier": 0.2080},
    "ITF|Hard (I)": {"a": -0.002100, "b": 1.259200, "n": 43165, "brier": 0.2158},
    "ITF|Grass": {"a": 0.000200, "b": 1.447800, "n": 3154, "brier": 0.1991},
    "ITF|Carpet": {"a": 0.000100, "b": 1.293000, "n": 17525, "brier": 0.2173},
}

# Map tournament categories to tier groups
CATEGORY_TO_TIER = {
    "Grand Slam (Men's)": "Grand Slam",
    "ATP 1000": "Masters 1000",
    "ATP 500": "ATP 500",
    "ATP 250": "ATP 250",
    "ATP Finals": "ATP 250",
    "Challengers": "Challenger", "Challenger 50": "Challenger",
    "Challenger 75": "Challenger", "Challenger 80": "Challenger",
    "Challenger 90": "Challenger", "Challenger 100": "Challenger",
    "Challenger 110": "Challenger", "Challenger 125": "Challenger",
    "Challenger 175": "Challenger", "Challenger Tour Finals": "Challenger",
    "M25": "ITF", "M15": "ITF",
}

# Normalize surface names
SURFACE_MAP = {
    "Clay": "Clay", "Hard": "Hard", "Hard Indoor": "Hard (I)",
    "Grass": "Grass", "Carpet": "Carpet",
    "Clay Indoor": "Clay", "Synthetic": "Hard", "Play": "Hard",
}


class WinProbabilityModel:
    """
    Win probability model for tennis matches.
    
    Uses logistic regression on log(opponent_rank / player_rank),
    with tier×surface-specific coefficients.
    """
    
    def __init__(self, params=None, clip_range=(0.03, 0.97)):
        self.params = params or MODEL_PARAMS
        self.clip_lo, self.clip_hi = clip_range
    
    def _get_model(self, tier, surface):
        """Look up best available model using hierarchy."""
        # Normalize
        tier = CATEGORY_TO_TIER.get(tier, tier)
        surface = SURFACE_MAP.get(surface, surface)
        
        # Level 1: Tier×Surface
        key = f"{tier}|{surface}"
        if key in self.params and self.params[key].get('n', 0) >= 500:
            return self.params[key], key
        
        # Level 2: Tier-only
        key = f"tier_{tier}"
        if key in self.params:
            return self.params[key], key
        
        # Level 3: Global
        return self.params['global'], 'global'
    
    def predict(self, player_rank, opponent_rank, surface='Hard', tier='Challenger'):
        """
        Predict P(player wins).
        
        Args:
            player_rank: Player's current ranking (int)
            opponent_rank: Opponent's current ranking (int)
            surface: Court surface ('Clay', 'Hard', 'Hard Indoor', 'Grass', 'Carpet')
            tier: Tournament tier or category name
        
        Returns:
            float: Probability of player winning [clip_lo, clip_hi]
        """
        if player_rank <= 0 or opponent_rank <= 0:
            raise ValueError("Rankings must be positive")
        
        model, _ = self._get_model(tier, surface)
        lrr = math.log(opponent_rank / player_rank)
        z = model['a'] + model['b'] * lrr
        p = 1 / (1 + math.exp(-max(-30, min(30, z))))
        
        return max(self.clip_lo, min(self.clip_hi, p))
    
    def predict_match(self, player_rank, opponent_rank, surface='Hard', tier='Challenger'):
        """
        Predict match outcome with details.
        
        Returns dict with probabilities, model used, and rank info.
        """
        model, model_key = self._get_model(tier, surface)
        lrr = math.log(opponent_rank / player_rank)
        z = model['a'] + model['b'] * lrr
        p_raw = 1 / (1 + math.exp(-max(-30, min(30, z))))
        p_clipped = max(self.clip_lo, min(self.clip_hi, p_raw))
        
        return {
            'p_win': p_clipped,
            'p_win_raw': p_raw,
            'log_rank_ratio': lrr,
            'model_used': model_key,
            'model_params': model,
            'player_rank': player_rank,
            'opponent_rank': opponent_rank,
        }
    
    def simulate_tournament(self, player_rank, draw_ranks, surface='Hard', 
                             tier='Challenger', n_sims=10000, seed=None):
        """
        Monte Carlo simulation of tournament progression.
        
        Simulates a single-elimination bracket where the player faces
        opponents drawn from draw_ranks in each round.
        
        Args:
            player_rank: Player's ranking
            draw_ranks: List of opponent rankings in the draw (used to 
                       randomly assign opponents per round)
            surface: Court surface
            tier: Tournament tier
            n_sims: Number of simulations
            seed: Random seed
        
        Returns:
            dict with round-reach probabilities
        """
        import random
        if seed is not None:
            random.seed(seed)
        
        n_rounds = 0
        n = len(draw_ranks)
        while (1 << n_rounds) < n:
            n_rounds += 1
        
        round_names = ['R1', 'R2', 'R3', 'R4', 'QF', 'SF', 'F', 'W']
        # Map based on draw size
        if n <= 32:
            labels = ['R1', 'R2', 'QF', 'SF', 'F', 'W'][:n_rounds + 1]
        elif n <= 64:
            labels = ['R1', 'R2', 'R3', 'QF', 'SF', 'F', 'W'][:n_rounds + 1]
        else:
            labels = ['R1', 'R2', 'R3', 'R4', 'QF', 'SF', 'F', 'W'][:n_rounds + 1]
        
        round_counts = {label: 0 for label in labels}
        round_counts[labels[0]] = n_sims  # Always enters R1
        
        for _ in range(n_sims):
            # Simulate each round
            for r in range(1, len(labels)):
                # Pick a random opponent from the draw
                opp_rank = random.choice(draw_ranks)
                p_win = self.predict(player_rank, opp_rank, surface, tier)
                
                if random.random() < p_win:
                    round_counts[labels[r]] += 1
                else:
                    break
        
        # Convert to probabilities
        round_probs = {label: count / n_sims for label, count in round_counts.items()}
        
        return round_probs
    
    def expected_value(self, player_rank, draw_ranks, prize_money, 
                       ranking_points=None, entry_cost=0,
                       surface='Hard', tier='Challenger', n_sims=10000, seed=None):
        """
        Calculate expected prize money and ranking points for a tournament.
        
        Args:
            player_rank: Player's ranking
            draw_ranks: List of opponent rankings in the draw
            prize_money: Dict mapping round labels to prize money
                        e.g. {'R1': 500, 'QF': 2000, 'SF': 5000, 'F': 10000, 'W': 20000}
            ranking_points: Dict mapping round labels to ranking points (optional)
            entry_cost: Cost to enter (entry fee + travel + accommodation)
            surface: Court surface
            tier: Tournament tier
            n_sims: Number of simulations
            seed: Random seed
        
        Returns:
            dict with expected prize money, expected points, ROI, round probabilities
        """
        round_probs = self.simulate_tournament(
            player_rank, draw_ranks, surface, tier, n_sims, seed)
        
        # Expected prize = sum(P(reaching round) * prize for that round)
        # Note: prize is for reaching that round, not for exiting at that round
        # P(exit at round R) = P(reach R) - P(reach R+1)
        rounds = list(round_probs.keys())
        
        exp_prize = 0.0
        exp_points = 0.0
        
        for i, r in enumerate(rounds):
            p_reach = round_probs[r]
            p_next = round_probs[rounds[i+1]] if i + 1 < len(rounds) else 0.0
            p_exit_here = p_reach - p_next
            
            if r in prize_money:
                exp_prize += p_exit_here * prize_money[r]
            if ranking_points and r in ranking_points:
                exp_points += p_exit_here * ranking_points[r]
        
        roi = (exp_prize - entry_cost) / entry_cost if entry_cost > 0 else float('inf')
        
        return {
            'expected_prize': round(exp_prize, 2),
            'expected_points': round(exp_points, 2) if ranking_points else None,
            'entry_cost': entry_cost,
            'expected_profit': round(exp_prize - entry_cost, 2),
            'roi': round(roi, 4),
            'round_probs': round_probs,
        }

    def summary(self):
        """Print model summary."""
        print("Win Probability Model Summary")
        print("=" * 50)
        print(f"Total models: {len(self.params)}")
        print(f"Clip range: [{self.clip_lo}, {self.clip_hi}]")
        print(f"\nTier coefficients (b = predictiveness of rankings):")
        print(f"  {'Tier':<20s} {'b':>6s}  {'Interpretation'}")
        print(f"  {'-'*55}")
        for tier in ['Grand Slam','Masters 1000','ATP 500','ATP 250','Challenger','ITF']:
            key = f'tier_{tier}'
            if key in self.params:
                b = self.params[key]['b']
                interp = "very predictable" if b > 1.2 else \
                         "predictable" if b > 0.9 else \
                         "moderate" if b > 0.7 else "upset-prone"
                print(f"  {tier:<20s} {b:>6.3f}  {interp}")


# =========================================================================
# Demo / test
# =========================================================================
if __name__ == '__main__':
    model = WinProbabilityModel()
    model.summary()
    
    print("\n\nExample predictions:")
    print("=" * 70)
    scenarios = [
        (150, 45, 'Clay', 'Challenger', 'Rank 150 vs 45 on clay at Challenger'),
        (150, 45, 'Hard', 'ATP 250', 'Rank 150 vs 45 on hard at ATP 250'),
        (300, 300, 'Clay', 'ITF', 'Equal rank 300 on clay at ITF'),
        (80, 200, 'Hard', 'Challenger', 'Rank 80 vs 200 on hard at Challenger'),
        (50, 10, 'Grass', 'Grand Slam', 'Rank 50 vs 10 on grass at Grand Slam'),
        (500, 100, 'Clay', 'Challenger', 'Rank 500 vs 100 on clay at Challenger'),
    ]
    
    for pr, opr, surf, tier, desc in scenarios:
        p = model.predict(pr, opr, surf, tier)
        print(f"  {desc:>50s}: P(win) = {100*p:.1f}%")
    
    # Monte Carlo example
    print("\n\nMonte Carlo Tournament Simulation:")
    print("=" * 70)
    print("Scenario: Rank 200 player entering a Challenger on Clay")
    print("Draw: 32 players, ranks sampled around 100-500")
    
    import random
    random.seed(42)
    draw = [random.randint(80, 500) for _ in range(32)]
    
    prize = {'R1': 480, 'R2': 960, 'QF': 2280, 'SF': 4440, 'F': 7680, 'W': 14400}
    points = {'R1': 0, 'R2': 6, 'QF': 12, 'SF': 25, 'F': 40, 'W': 80}
    
    ev = model.expected_value(
        player_rank=200, draw_ranks=draw, prize_money=prize,
        ranking_points=points, entry_cost=1500,
        surface='Clay', tier='Challenger', n_sims=50000, seed=42
    )
    
    print(f"\n  Round probabilities:")
    for r, p in ev['round_probs'].items():
        pm = prize.get(r, 0)
        print(f"    {r:>3s}: {100*p:>6.1f}%  (prize: ${pm:,})")
    
    print(f"\n  Expected prize:  ${ev['expected_prize']:,.0f}")
    print(f"  Expected points: {ev['expected_points']:.1f}")
    print(f"  Entry cost:      ${ev['entry_cost']:,}")
    print(f"  Expected profit: ${ev['expected_profit']:,.0f}")
    print(f"  ROI:             {100*ev['roi']:.1f}%")
