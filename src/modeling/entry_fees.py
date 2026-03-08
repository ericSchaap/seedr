"""
Tennis Tournament Optimizer - Entry Fees & Tournament Costs
=============================================================

Models the full out-of-pocket costs for entering a tournament, beyond
just travel. Based on official ATP/ITF rules (2025 season).

Key rules:
  - ATP Tour & ATP Challenger Tour: NO entry fees
  - ITF World Tennis Tour (M15/M25): max $40 entry fee
  - Challenger tournaments: FREE hotel accommodation (mandatory per ATP rules)
  - ITF tournaments: accommodation NOT provided (player pays)
  - Grand Slams & ATP events: various hospitality provisions

This module adjusts the total cost estimate from the travel model by
adding entry fees and adjusting for accommodation provisions.

Usage:
    from entry_fees import get_entry_fee, get_accommodation_status, get_total_tournament_cost

    fee = get_entry_fee("M25")                    # $40
    fee = get_entry_fee("Challenger 75")           # $0
    accom = get_accommodation_status("Challenger 75")  # 'provided'
    
    cost = get_total_tournament_cost("Challenger 75", travel_cost=1500)  # Adjusted
"""


# =========================================================================
# ENTRY FEES BY CATEGORY
# =========================================================================
# Source: ITF rules cap entry fees at $40 for singles+doubles
# ATP Tour and Challenger Tour have zero entry fees

ENTRY_FEES = {
    # ATP Tour: no entry fees
    "Grand Slam (Men's)": 0,
    "Grand Slam (Women's)": 0,
    "ATP 1000": 0,
    "WTA 1000": 0,
    "WTA 1000 (5)": 0,
    "ATP 500": 0,
    "WTA 500": 0,
    "ATP 250": 0,
    "WTA 250": 0,
    "WTA 125": 0,
    "ATP Finals": 0,
    "Year End Championships": 0,

    # Challenger Tour: no entry fees
    "Challenger 175": 0,
    "Challenger 125": 0,
    "Challenger 110": 0,
    "Challenger 100": 0,
    "Challenger 90": 0,
    "Challenger 80": 0,
    "Challenger 75": 0,
    "Challenger 50": 0,

    # ITF World Tennis Tour: max $40
    "M25": 40,
    "M15": 40,
    "W100": 40,
    "W80": 40,
    "W75": 40,
    "W50": 40,
    "W35": 40,
    "W25": 40,
    "W15": 40,
    "W10": 40,
}


# =========================================================================
# ACCOMMODATION STATUS
# =========================================================================
# 'provided' = tournament pays hotel (Challengers: mandatory per ATP rules)
# 'partial'  = some provisions (Grand Slams provide for qualifiers)
# 'none'     = player pays everything (ITF events)

ACCOMMODATION = {
    "Grand Slam (Men's)": "partial",    # Housing for qualifiers
    "Grand Slam (Women's)": "partial",
    "ATP 1000": "partial",
    "WTA 1000": "partial",
    "ATP 500": "none",                  # Varies by tournament
    "WTA 500": "none",
    "ATP 250": "none",
    "WTA 250": "none",
    "Challenger 175": "provided",       # Mandatory per ATP rules
    "Challenger 125": "provided",
    "Challenger 110": "provided",
    "Challenger 100": "provided",
    "Challenger 90": "provided",
    "Challenger 80": "provided",
    "Challenger 75": "provided",
    "Challenger 50": "provided",
    "M25": "none",
    "M15": "none",
}

# Accommodation cost per night (average, used when player pays)
NIGHTLY_ACCOMMODATION = {
    'Europe': 80,
    'North America': 100,
    'Asia': 50,
    'South America': 45,
    'Africa': 40,
    'Oceania': 90,
}

TYPICAL_NIGHTS = {
    "Grand Slam (Men's)": 10,   # 2-week event
    "ATP 1000": 6,
    "ATP 500": 5,
    "ATP 250": 5,
    "Challenger": 5,            # But accommodation is provided
    "ITF": 5,                   # Player pays
}


def get_entry_fee(category):
    """
    Get the entry fee for a tournament category.

    Args:
        category: Tournament category string

    Returns:
        int: Entry fee in USD
    """
    if not isinstance(category, str):
        return 0

    # Exact match
    if category in ENTRY_FEES:
        return ENTRY_FEES[category]

    # Partial match
    for key, fee in ENTRY_FEES.items():
        if key in category:
            return fee

    # Default: ITF-level events have fees, everything else doesn't
    cat_lower = category.lower()
    if 'm15' in cat_lower or 'm25' in cat_lower or cat_lower.startswith('w'):
        return 40
    return 0


def get_accommodation_status(category):
    """
    Get whether accommodation is provided by the tournament.

    Returns:
        str: 'provided', 'partial', or 'none'
    """
    if not isinstance(category, str):
        return 'none'

    if category in ACCOMMODATION:
        return ACCOMMODATION[category]

    for key, status in ACCOMMODATION.items():
        if key in category:
            return status

    if 'Challenger' in category:
        return 'provided'
    return 'none'


def get_accommodation_cost(category, continent='Europe'):
    """
    Estimate accommodation cost when the player has to pay.
    Returns 0 if accommodation is provided by the tournament.

    Args:
        category: Tournament category
        continent: Tournament location continent

    Returns:
        int: Estimated accommodation cost in USD
    """
    status = get_accommodation_status(category)
    if status == 'provided':
        return 0

    nightly = NIGHTLY_ACCOMMODATION.get(continent, 70)

    # Determine typical nights
    cat_lower = (category or '').lower()
    if 'grand slam' in cat_lower:
        nights = 10
    elif '1000' in cat_lower:
        nights = 6
    elif '500' in cat_lower or '250' in cat_lower:
        nights = 5
    else:
        nights = 5  # ITF default

    if status == 'partial':
        nights = max(0, nights - 3)  # Tournament covers ~3 nights

    return nightly * nights


def get_total_tournament_cost(category, travel_cost, continent='Europe'):
    """
    Calculate total out-of-pocket cost for entering a tournament.
    Combines travel, entry fee, and accommodation.

    Args:
        category: Tournament category
        travel_cost: Base travel cost from TravelCostModel
        continent: Tournament location continent

    Returns:
        dict with cost breakdown
    """
    entry_fee = get_entry_fee(category)
    accom_cost = get_accommodation_cost(category, continent)
    accom_status = get_accommodation_status(category)

    # Adjust travel cost for Challengers: accommodation is provided,
    # so the travel_cost model's hotel component is overstated
    if accom_status == 'provided':
        # Reduce travel cost by ~40% (hotel portion)
        adjusted_travel = int(travel_cost * 0.6)
    else:
        adjusted_travel = travel_cost

    total = adjusted_travel + entry_fee + accom_cost

    return {
        'total_cost': total,
        'travel': adjusted_travel,
        'entry_fee': entry_fee,
        'accommodation': accom_cost,
        'accommodation_status': accom_status,
    }


# =========================================================================
# DEMO
# =========================================================================
if __name__ == '__main__':
    print("Entry Fees & Tournament Costs")
    print("=" * 60)

    from travel_costs import TravelCostModel, COUNTRY_CONTINENT

    model = TravelCostModel(player_country='FRA')

    test_cases = [
        ("Challenger 75", "ESP"),
        ("Challenger 125", "ITA"),
        ("M25", "TUN"),
        ("M15", "FRA"),
        ("ATP 250", "GER"),
        ("ATP 1000", "ESP"),
        ("Grand Slam (Men's)", "FRA"),
    ]

    print(f"\n{'Category':<22s} {'Country':>5s} {'Travel':>8s} {'Entry':>6s} "
          f"{'Accom':>7s} {'Total':>8s} {'Hotel':>10s}")
    print("-" * 75)

    for cat, country in test_cases:
        travel = model.estimate_cost(country)
        continent = COUNTRY_CONTINENT.get(country, 'Europe')
        costs = get_total_tournament_cost(cat, travel, continent)

        print(f"  {cat:<20s} {country:>5s} ${travel:>6,} ${costs['entry_fee']:>4} "
              f"${costs['accommodation']:>5,} ${costs['total_cost']:>6,} "
              f"{costs['accommodation_status']:>10s}")
