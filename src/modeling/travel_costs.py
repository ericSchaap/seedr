"""
Seedr - Travel Cost Model v2 (EU Commission Data)
====================================================

Distance-based cost model using official EU reimbursement rates (July 2024),
derived from 131,000+ real flight tickets across 9,000+ connections.

Three cost components, each estimated separately:
  1. FLIGHTS - great-circle distance -> EU cost band -> budget discount
  2. HOTELS - per-night country rate x tournament duration (free at Challengers)
  3. ENTRY FEES - from entry_fees module (unchanged)

Player overhead (coach, physio, food) is handled in the app layer, not here.

Source: Commission Decision C(2024) 5405, Annex
        https://ec.europa.eu/info/funding-tenders/opportunities/docs/
        2021-2027/common/guidance/unit-cost-decision-travel_en.pdf

Usage:
    from travel_costs import TravelCostModel, COUNTRY_CONTINENT
    model = TravelCostModel(player_country='FRA', home_city='Paris')
    info = model.get_schedule_travel_info(schedule)
"""

import math


# ==============================================================================
# COUNTRY -> CONTINENT (preserved for optimizer geographic constraints)
# ==============================================================================

COUNTRY_CONTINENT = {
    'ALB': 'Europe', 'AND': 'Europe', 'ARM': 'Europe', 'AUT': 'Europe',
    'AZE': 'Europe', 'BEL': 'Europe', 'BIH': 'Europe', 'BLR': 'Europe',
    'BUL': 'Europe', 'CRO': 'Europe', 'CYP': 'Europe', 'CZE': 'Europe',
    'DEN': 'Europe', 'ESP': 'Europe', 'EST': 'Europe', 'FIN': 'Europe',
    'FRA': 'Europe', 'GBR': 'Europe', 'GEO': 'Europe', 'GER': 'Europe',
    'GRE': 'Europe', 'HUN': 'Europe', 'IRL': 'Europe', 'ISR': 'Europe',
    'ITA': 'Europe', 'KOS': 'Europe', 'LAT': 'Europe', 'LIT': 'Europe',
    'LUX': 'Europe', 'MDA': 'Europe', 'MKD': 'Europe', 'MLT': 'Europe',
    'MNE': 'Europe', 'MON': 'Europe', 'NED': 'Europe', 'NOR': 'Europe',
    'POL': 'Europe', 'POR': 'Europe', 'ROU': 'Europe', 'RUS': 'Europe',
    'SLO': 'Europe', 'SRB': 'Europe', 'SUI': 'Europe', 'SVK': 'Europe',
    'SWE': 'Europe', 'TUR': 'Europe', 'UKR': 'Europe',
    'CAN': 'North America', 'USA': 'North America', 'MEX': 'North America',
    'DOM': 'North America', 'GUA': 'North America', 'CRC': 'North America',
    'JAM': 'North America', 'TTO': 'North America', 'PUR': 'North America',
    'CUB': 'North America', 'BAH': 'North America',
    'ARG': 'South America', 'BOL': 'South America', 'BRA': 'South America',
    'CHI': 'South America', 'COL': 'South America', 'ECU': 'South America',
    'PAR': 'South America', 'PER': 'South America', 'URU': 'South America',
    'VEN': 'South America',
    'BRN': 'Asia', 'CHN': 'Asia', 'HKG': 'Asia', 'IND': 'Asia',
    'INA': 'Asia', 'JPN': 'Asia', 'KAZ': 'Asia', 'KOR': 'Asia',
    'KGZ': 'Asia', 'MAS': 'Asia', 'PHI': 'Asia', 'QAT': 'Asia',
    'SGP': 'Asia', 'SRI': 'Asia', 'THA': 'Asia', 'TPE': 'Asia',
    'UAE': 'Asia', 'UZB': 'Asia', 'VIE': 'Asia',
    'ANG': 'Africa', 'CIV': 'Africa', 'EGY': 'Africa', 'KEN': 'Africa',
    'MAR': 'Africa', 'NGR': 'Africa', 'RSA': 'Africa', 'TUN': 'Africa',
    'UGA': 'Africa',
    'AUS': 'Oceania', 'NZL': 'Oceania',
}

ADJACENT_REGIONS = {
    'Europe': ['Africa'], 'Africa': ['Europe'],
    'North America': ['South America'], 'South America': ['North America'],
}


# ==============================================================================
# EU COMMISSION 2024 FLIGHT COST BANDS (EUR, return trip)
# Source: C(2024) 5405, Table 5.1 -- 131,000+ real tickets from Kayak
# ==============================================================================

EU_FLIGHT_BANDS = [
    (600, 340), (1600, 365), (2500, 429), (3500, 541),
    (4500, 743), (6000, 857), (7500, 1021), (10000, 1250), (99999, 1595),
]

# Intra-country land travel (EUR, return trip, 50-399 km)
EU_LAND_TRAVEL = {
    'AUT': 65, 'BEL': 58, 'BUL': 13, 'CZE': 28, 'GER': 64,
    'DEN': 83, 'EST': 20, 'GRE': 39, 'ESP': 53, 'FIN': 38,
    'FRA': 72, 'CRO': 39, 'HUN': 29, 'IRL': 41, 'ITA': 52,
    'LIT': 29, 'LAT': 17, 'NED': 61, 'POL': 21, 'POR': 44,
    'ROU': 18, 'SWE': 56, 'SLO': 35, 'SVK': 22,
    'GBR': 70, 'SUI': 85, 'NOR': 80, 'SRB': 20, 'BIH': 15,
    'MKD': 15, 'MNE': 18, 'ALB': 12, 'KOS': 12, 'GEO': 15,
    'TUR': 25, 'USA': 60, 'CAN': 55, 'AUS': 60, 'JPN': 70,
    'ARG': 20, 'BRA': 25, 'MEX': 20, 'CHN': 25, 'IND': 12,
    'TUN': 10, 'EGY': 10, 'MAR': 15, 'RSA': 20, 'KOR': 30,
    'THA': 10, 'KAZ': 15, 'COL': 12, 'CHI': 20, 'PER': 12,
}

# Budget discount: players book LCCs, advance purchase, basic economy.
# EU rates = average business/leisure mix. Players pay ~65% of that.
PLAYER_FLIGHT_DISCOUNT = 0.65
PLAYER_LAND_DISCOUNT = 1.0


# ==============================================================================
# HOTEL COSTS (EUR per night) -- EU Commission 2024 official ceilings
# Source: C(2024) 5405, Table 5.5
# Players in budget 2-3 star hotels pay ~55% of the business ceiling.
# ==============================================================================

EU_HOTEL_PER_NIGHT = {
    'ALB': 101, 'AUT': 126, 'BEL': 137, 'BIH': 90, 'BUL': 110,
    'CRO': 104, 'CYP': 120, 'CZE': 107, 'DEN': 158, 'EST': 107,
    'FIN': 146, 'FRA': 166, 'GER': 119, 'GRE': 107, 'HUN': 105,
    'IRL': 139, 'ITA': 114, 'KOS': 92, 'LAT': 95, 'LIT': 94,
    'LUX': 163, 'MLT': 141, 'MNE': 98, 'MKD': 95, 'NED': 133,
    'POL': 103, 'POR': 109, 'ROU': 109, 'SRB': 105, 'SVK': 98,
    'SLO': 113, 'ESP': 117, 'SWE': 158,
    'GBR': 151, 'NOR': 145, 'SUI': 178, 'TUR': 116, 'UKR': 122,
    'GEO': 134, 'ARM': 115, 'AZE': 136, 'BLR': 108, 'MDA': 133,
    'ISR': 187, 'EGY': 152, 'TUN': 99, 'MAR': 129, 'MON': 166,
    'USA': 160, 'CAN': 140, 'AUS': 140, 'NZL': 120,
    'JPN': 130, 'KOR': 110, 'CHN': 90, 'IND': 70, 'THA': 60,
    'SGP': 150, 'HKG': 160, 'UAE': 140, 'QAT': 150,
    'ARG': 70, 'BRA': 80, 'CHI': 80, 'COL': 60, 'PER': 55,
    'MEX': 70, 'RSA': 70, 'KEN': 60, 'KAZ': 70,
    'INA': 50, 'MAS': 55, 'PHI': 50, 'VIE': 45, 'SRI': 50,
    'UZB': 50, 'KGZ': 45, 'DOM': 65, 'CRC': 65,
}

PLAYER_HOTEL_DISCOUNT = 0.55

TOURNAMENT_NIGHTS = {
    'M15': 5, 'M25': 5, 'Challenger': 6,
    'ATP 250': 7, 'ATP 500': 7, 'ATP 1000': 8,
    'Grand Slam': 10, 'ATP Finals': 7,
}

FREE_ACCOMMODATION_CATEGORIES = {'Challenger', '+H'}


# ==============================================================================
# CITY COORDINATES (lat, lon) for distance computation
# ==============================================================================

CITY_COORDS = {
    'Melbourne': (-37.81, 144.96), 'Paris': (48.86, 2.35),
    'London': (51.51, -0.13), 'New York': (40.71, -74.01),
    'Wimbledon': (51.43, -0.19), 'Roland Garros': (48.85, 2.25),
    'Madrid': (40.42, -3.70), 'Rome': (41.90, 12.50),
    'Barcelona': (41.39, 2.17), 'Monte Carlo': (43.74, 7.43),
    'Monte-Carlo': (43.74, 7.43), 'Monaco': (43.74, 7.43),
    'Miami': (25.76, -80.19), 'Indian Wells': (33.72, -116.31),
    'Cincinnati': (39.10, -84.51), 'Shanghai': (31.23, 121.47),
    'Toronto': (43.65, -79.38), 'Montreal': (45.50, -73.57),
    'Hamburg': (53.55, 9.99), 'Vienna': (48.21, 16.37),
    'Bordeaux': (44.84, -0.58), 'Lyon': (45.76, 4.84),
    'Marseille': (43.30, 5.37), 'Bucharest': (44.43, 26.10),
    'Prague': (50.08, 14.44), 'Budapest': (47.50, 19.04),
    'Zagreb': (45.81, 15.98), 'Split': (43.51, 16.44),
    'Belgrade': (44.79, 20.47), 'Istanbul': (41.01, 28.98),
    'Antalya': (36.90, 30.70), 'Munich': (48.14, 11.58),
    'Stuttgart': (48.78, 9.18), 'Augsburg': (48.37, 10.90),
    'Lisbon': (38.72, -9.14), 'Porto': (41.15, -8.61),
    'Oeiras': (38.69, -9.31), 'Estoril': (38.71, -9.40),
    'Milan': (45.46, 9.19), 'Turin': (45.07, 7.69),
    'Florence': (43.77, 11.25), 'Barletta': (41.32, 16.28),
    'Vicenza': (45.55, 11.55), 'Bergamo': (45.70, 9.67),
    'Cagliari': (39.22, 9.12), 'Santa Margherita Di Pula': (39.10, 9.02),
    'Francavilla al Mare': (42.42, 14.29), 'Reggio Emilia': (44.70, 10.63),
    'Amsterdam': (52.37, 4.90), 'Geneva': (46.20, 6.14),
    'Basel': (47.56, 7.59), 'Zurich': (47.38, 8.54),
    'Bratislava': (48.15, 17.11), 'Ostrava': (49.82, 18.26),
    'Skopje': (42.00, 21.43), 'Sarajevo': (43.86, 18.41),
    'Tbilisi': (41.69, 44.80), 'Kachreti': (41.61, 45.82),
    'Warsaw': (52.23, 21.01), 'Athens': (37.98, 23.73),
    'Monastir': (35.78, 10.83), 'Tunis': (36.81, 10.18),
    'Casablanca': (33.57, -7.59), 'Rabat': (34.02, -6.84),
    'Marrakech': (31.63, -8.01), 'Marbella': (36.51, -4.88),
    'Reus': (41.15, 1.11), 'Prostejov': (49.47, 17.11),
    'Mauthausen': (48.25, 14.52), 'Grasse': (43.66, 6.92),
    'Buenos Aires': (-34.60, -58.38), 'Sao Paulo': (-23.55, -46.63),
    'Santiago': (-33.45, -70.67), 'Bogota': (4.71, -74.07),
    'Lima': (-12.05, -77.04), 'Florianopolis': (-27.60, -48.55),
    'Porto Alegre': (-30.03, -51.23), 'Tucuman': (-26.81, -65.22),
    'Tokyo': (35.68, 139.69), 'Beijing': (39.91, 116.40),
    'Seoul': (37.57, 126.98), 'Singapore': (1.35, 103.82),
    'Bangkok': (13.76, 100.50), 'Delhi': (28.61, 77.21),
    'Dubai': (25.20, 55.27), 'Doha': (25.29, 51.53),
    'Johannesburg': (-26.20, 28.05), 'Cairo': (30.04, 31.24),
    'Sydney': (-33.87, 151.21),
}

COUNTRY_CAPITAL_COORDS = {
    'ALB': (41.33, 19.82), 'AND': (42.51, 1.52), 'ARM': (40.18, 44.51),
    'AUT': (48.21, 16.37), 'AZE': (40.41, 49.87), 'BEL': (50.85, 4.35),
    'BIH': (43.86, 18.41), 'BLR': (53.90, 27.57), 'BUL': (42.70, 23.32),
    'CRO': (45.81, 15.98), 'CYP': (35.17, 33.36), 'CZE': (50.08, 14.44),
    'DEN': (55.68, 12.57), 'ESP': (40.42, -3.70), 'EST': (59.44, 24.75),
    'FIN': (60.17, 24.94), 'FRA': (48.86, 2.35), 'GBR': (51.51, -0.13),
    'GEO': (41.69, 44.80), 'GER': (52.52, 13.41), 'GRE': (37.98, 23.73),
    'HUN': (47.50, 19.04), 'IRL': (53.35, -6.26), 'ISR': (31.77, 35.22),
    'ITA': (41.90, 12.50), 'KOS': (42.66, 21.17), 'KAZ': (51.17, 71.45),
    'LAT': (56.95, 24.11), 'LIT': (54.69, 25.28), 'LUX': (49.61, 6.13),
    'MDA': (47.01, 28.86), 'MKD': (42.00, 21.43), 'MLT': (35.90, 14.51),
    'MNE': (42.44, 19.26), 'MON': (43.74, 7.43), 'MAR': (34.02, -6.84),
    'NED': (52.37, 4.90), 'NOR': (59.91, 10.75), 'POL': (52.23, 21.01),
    'POR': (38.72, -9.14), 'ROU': (44.43, 26.10), 'RUS': (55.76, 37.62),
    'SLO': (46.05, 14.51), 'SRB': (44.79, 20.47), 'SUI': (46.95, 7.45),
    'SVK': (48.15, 17.11), 'SWE': (59.33, 18.07), 'TUR': (39.93, 32.85),
    'TUN': (36.81, 10.18), 'UKR': (50.45, 30.52),
    'USA': (38.91, -77.04), 'CAN': (45.42, -75.70), 'MEX': (19.43, -99.13),
    'ARG': (-34.60, -58.38), 'BRA': (-15.79, -47.88), 'CHI': (-33.45, -70.67),
    'COL': (4.71, -74.07), 'PER': (-12.05, -77.04), 'ECU': (-0.18, -78.47),
    'URU': (-34.88, -56.17), 'AUS': (-33.87, 151.21), 'NZL': (-41.29, 174.78),
    'JPN': (35.68, 139.69), 'KOR': (37.57, 126.98), 'CHN': (39.91, 116.40),
    'IND': (28.61, 77.21), 'THA': (13.76, 100.50), 'SGP': (1.35, 103.82),
    'INA': (-6.21, 106.85), 'MAS': (3.14, 101.69), 'HKG': (22.32, 114.17),
    'UAE': (25.20, 55.27), 'QAT': (25.29, 51.53),
    'EGY': (30.04, 31.24), 'RSA': (-26.20, 28.05), 'KEN': (-1.29, 36.82),
    'TUN': (36.81, 10.18), 'MAR': (34.02, -6.84),
    'KGZ': (42.87, 74.59), 'UZB': (41.30, 69.28), 'KAZ': (51.17, 71.45),
    'PHI': (14.60, 120.98), 'VIE': (21.03, 105.85), 'TPE': (25.03, 121.57),
    'SRI': (6.93, 79.85), 'BOL': (-16.50, -68.15), 'PAR': (-25.26, -57.58),
    'VEN': (10.49, -66.90), 'DOM': (18.47, -69.90), 'CRC': (9.93, -84.09),
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points in km."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def get_city_coords(city_name, country_code=None):
    """Look up coordinates for a city. Falls back to country capital."""
    if not city_name:
        return COUNTRY_CAPITAL_COORDS.get(country_code) if country_code else None
    if city_name in CITY_COORDS:
        return CITY_COORDS[city_name]
    cl = city_name.lower()
    for key, coords in CITY_COORDS.items():
        if key.lower() == cl:
            return coords
    for key, coords in CITY_COORDS.items():
        if cl in key.lower() or key.lower() in cl:
            return coords
    return COUNTRY_CAPITAL_COORDS.get(country_code) if country_code else None


def extract_city_from_tournament(name):
    """Extract city name from tournament name string."""
    name = name or ''
    for prefix in ['M15+H ', 'M25+H ', 'M15 ', 'M25 ']:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    parts = name.rsplit(' ', 1)
    if len(parts) == 2 and parts[1].isdigit():
        name = parts[0]
    return name.replace(' Challenger', '').strip()


def flight_cost_eur(distance_km):
    """One-way flight cost in EUR from EU distance bands + player discount."""
    if distance_km < 50:
        return 0
    if distance_km < 400:
        return round(35 * PLAYER_FLIGHT_DISCOUNT)
    for max_km, return_cost in EU_FLIGHT_BANDS:
        if distance_km <= max_km:
            return round((return_cost / 2) * PLAYER_FLIGHT_DISCOUNT)
    return round((EU_FLIGHT_BANDS[-1][1] / 2) * PLAYER_FLIGHT_DISCOUNT)


def land_travel_cost_eur(country_code):
    """One-way intra-country land travel cost."""
    base = EU_LAND_TRAVEL.get(country_code, 30)
    return round((base / 2) * PLAYER_LAND_DISCOUNT)


def hotel_nightly_rate(country_code):
    """Budget hotel cost per night for a country."""
    base = EU_HOTEL_PER_NIGHT.get(country_code, 100)
    return round(base * PLAYER_HOTEL_DISCOUNT)


def tournament_nights(category):
    """Typical number of nights for a tournament category."""
    if not isinstance(category, str):
        return 5
    for key, nights in TOURNAMENT_NIGHTS.items():
        if key in category:
            return nights
    if 'Grand Slam' in category:
        return 10
    return 5


def is_free_accommodation(category):
    """Check if tournament provides free accommodation."""
    if not isinstance(category, str):
        return False
    return any(marker in category for marker in FREE_ACCOMMODATION_CATEGORIES)


# ==============================================================================
# MAIN MODEL CLASS
# ==============================================================================

class TravelCostModel:
    """
    Distance-based travel cost model using EU Commission 2024 data.
    Backward-compatible interface with the old 3-tier model.
    """

    def __init__(self, player_country='FRA', home_city=None):
        self.player_country = player_country.upper()
        self.player_continent = COUNTRY_CONTINENT.get(self.player_country, 'Europe')
        self.home_city = home_city
        if home_city:
            self._home_coords = get_city_coords(home_city, self.player_country)
        else:
            self._home_coords = COUNTRY_CAPITAL_COORDS.get(self.player_country)

    def get_tier(self, tournament_country):
        """Determine travel tier (preserved for optimizer compatibility)."""
        if not tournament_country or not isinstance(tournament_country, str):
            return 'international'
        tc = tournament_country.upper()
        if tc == self.player_country:
            return 'national'
        tc_cont = COUNTRY_CONTINENT.get(tc, 'Unknown')
        if tc_cont == self.player_continent:
            return 'international'
        if tc_cont in ADJACENT_REGIONS.get(self.player_continent, []):
            return 'international'
        return 'intercontinental'

    def _get_distance(self, origin_coords, dest_city, dest_country):
        """Distance in km between origin and destination."""
        if origin_coords is None:
            tier = self.get_tier(dest_country)
            return {'national': 200, 'international': 1200,
                    'intercontinental': 5000}.get(tier, 1200)
        dest_coords = get_city_coords(dest_city, dest_country)
        if dest_coords is None:
            dest_coords = COUNTRY_CAPITAL_COORDS.get(dest_country)
        if dest_coords is None:
            tier = self.get_tier(dest_country)
            return {'national': 200, 'international': 1200,
                    'intercontinental': 5000}.get(tier, 1200)
        return haversine_km(origin_coords[0], origin_coords[1],
                            dest_coords[0], dest_coords[1])

    def estimate_tournament_cost(self, tournament, origin_coords=None):
        """Full cost breakdown for a single tournament."""
        if origin_coords is None:
            origin_coords = self._home_coords
        name = tournament.get('tournament_name', '')
        country = tournament.get('country', '')
        category = tournament.get('category', '')
        if not isinstance(country, str):
            country = ''
        dest_city = extract_city_from_tournament(name)
        distance = self._get_distance(origin_coords, dest_city, country)

        # Flight
        if distance < 400 and country == self.player_country:
            flight = land_travel_cost_eur(country)
        else:
            flight = flight_cost_eur(distance)

        # Hotel
        nights = tournament_nights(category)
        nightly = hotel_nightly_rate(country)
        if is_free_accommodation(category):
            hotel = 0
            hotel_saved = nightly * nights
        else:
            hotel = nightly * nights
            hotel_saved = 0

        # Entry fee
        try:
            from entry_fees import get_entry_fee
            entry = get_entry_fee(category)
        except ImportError:
            entry = 0

        return {
            'flight': flight, 'hotel': hotel, 'entry': entry,
            'total': flight + hotel + entry,
            'distance_km': round(distance), 'nights': nights,
            'nightly_rate': nightly, 'hotel_saved': hotel_saved,
            'free_accommodation': is_free_accommodation(category),
            'dest_city': dest_city, 'dest_country': country,
        }

    def estimate_cost(self, tournament_country):
        """Backward-compatible: estimate cost from country alone."""
        dummy = {'tournament_name': '', 'country': tournament_country,
                 'category': 'Challenger 75'}
        return self.estimate_tournament_cost(dummy)['total']

    def get_schedule_travel_info(self, schedule):
        """
        Full schedule costs with city-to-city sequential routing.
        Routes: home -> T1 -> T2 -> ... -> Tn (return home not included).
        """
        if not schedule:
            return {'total_cost': 0, 'per_tournament': [],
                    'tier_breakdown': {'national': 0, 'international': 0,
                                       'intercontinental': 0},
                    'avg_cost_per_tournament': 0}

        sorted_sched = sorted(schedule, key=lambda x: x[0])
        total_cost = 0
        details = []
        tier_counts = {'national': 0, 'international': 0, 'intercontinental': 0}
        current_coords = self._home_coords

        for week, tournament in sorted_sched:
            country = tournament.get('country', '')
            if not isinstance(country, str):
                country = ''
            tier = self.get_tier(country)
            tier_counts[tier] += 1

            cost_info = self.estimate_tournament_cost(
                tournament, origin_coords=current_coords)
            total_cost += cost_info['total']

            details.append({
                'week': week,
                'tournament': tournament.get('tournament_name', '?'),
                'country': country, 'tier': tier,
                'cost': cost_info['total'],
                'flight': cost_info['flight'],
                'hotel': cost_info['hotel'],
                'entry': cost_info['entry'],
                'distance_km': cost_info['distance_km'],
                'free_accommodation': cost_info['free_accommodation'],
            })

            dest_city = extract_city_from_tournament(
                tournament.get('tournament_name', ''))
            new_coords = get_city_coords(dest_city, country)
            if new_coords:
                current_coords = new_coords

        return {
            'total_cost': total_cost, 'per_tournament': details,
            'tier_breakdown': tier_counts,
            'avg_cost_per_tournament': round(
                total_cost / max(len(sorted_sched), 1)),
        }

    def get_full_schedule_cost(self, schedule, overhead_per_week=None):
        """
        Full schedule cost including travel, hotel, entry, AND player overhead.

        Args:
            schedule: list of (week, tournament_dict) tuples
            overhead_per_week: dict with weekly overhead costs, e.g.
                {'coach': 400, 'physio': 150, 'food': 200, 'other': 50}

        Returns:
            dict with total breakdown: flights, hotels, entry, overhead, grand_total
        """
        travel_info = self.get_schedule_travel_info(schedule)

        total_flights = sum(d['flight'] for d in travel_info['per_tournament'])
        total_hotels = sum(d['hotel'] for d in travel_info['per_tournament'])
        total_entry = sum(d['entry'] for d in travel_info['per_tournament'])

        # Overhead: applies per active week (weeks where a tournament is played)
        if overhead_per_week and any(v > 0 for v in overhead_per_week.values()):
            weekly_total = sum(overhead_per_week.values())
            active_weeks = len(set(w for w, _ in schedule))
            total_overhead = weekly_total * active_weeks
        else:
            weekly_total = 0
            active_weeks = 0
            total_overhead = 0

        grand_total = total_flights + total_hotels + total_entry + total_overhead

        return {
            'total_flights': round(total_flights),
            'total_hotels': round(total_hotels),
            'total_entry': round(total_entry),
            'total_overhead': round(total_overhead),
            'overhead_per_week': round(weekly_total),
            'active_weeks': active_weeks,
            'grand_total': round(grand_total),
            'per_tournament': travel_info['per_tournament'],
            'tier_breakdown': travel_info['tier_breakdown'],
        }
