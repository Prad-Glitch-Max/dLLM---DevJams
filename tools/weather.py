"""
Open-Meteo Live Weather Tool
============================
Integrates the Open-Meteo Free Weather API & Geocoding API.
No API key required.
"""

import re
import requests
from datetime import datetime
from typing import Dict, Any, Optional

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Comprehensive WMO Weather interpretation codes
WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail"
}

# Weather icons mapping for UI
WEATHER_ICONS = {
    0: "☀️",
    1: "🌤️",
    2: "⛅",
    3: "☁️",
    45: "🌫️",
    48: "🌫️",
    51: "🌦️",
    53: "🌦️",
    55: "🌧️",
    61: "🌧️",
    63: "🌧️",
    65: "⛈️",
    71: "🌨️",
    73: "❄️",
    75: "❄️",
    80: "🌦️",
    81: "🌧️",
    82: "⛈️",
    95: "⚡",
    96: "⛈️",
    99: "⛈️"
}


import difflib

# Known fallback cities for quick check and typo correction
KNOWN_CITIES = [
    "Chennai", "Vellore", "Bangalore", "Bengaluru", "Mumbai", "Delhi", "New Delhi",
    "Hyderabad", "Coimbatore", "Pune", "Kochi", "Kolkata", "Tokyo", "London",
    "New York", "San Francisco", "Paris", "Berlin", "Sydney", "Singapore",
    "Dubai", "Toronto", "Seattle", "Chicago", "Boston"
]

CITY_SYNONYMS = {
    "cheenai": "Chennai",
    "chenai": "Chennai",
    "madras": "Chennai",
    "banglore": "Bangalore",
    "bengaluru": "Bangalore",
    "bombay": "Mumbai",
    "calcutta": "Kolkata",
    "newdelhi": "New Delhi",
    "dilli": "Delhi"
}


def normalize_city_name(candidate: str) -> str:
    """Normalizes city name against known typos and fuzzy matches."""
    cand_clean = candidate.strip().lower()
    if cand_clean in CITY_SYNONYMS:
        return CITY_SYNONYMS[cand_clean]
    
    # Check exact match in known cities
    for city in KNOWN_CITIES:
        if city.lower() == cand_clean:
            return "Bangalore" if city.lower() == "bengaluru" else city

    # Fuzzy match with cutoff
    fuzzy = difflib.get_close_matches(candidate.strip().title(), KNOWN_CITIES, n=1, cutoff=0.6)
    if fuzzy:
        matched = fuzzy[0]
        return "Bangalore" if matched.lower() == "bengaluru" else matched

    return candidate.strip().title()


def extract_location(query: str) -> str:
    """
    Extracts location name from query using keyword, typo correction, and regex patterns.
    """
    query_clean = query.strip()
    
    # 1. Quick check for direct city names or known typos inside query
    query_words = re.findall(r"\b[a-zA-Z]+\b", query_clean.lower())
    for word in query_words:
        if word in CITY_SYNONYMS:
            return CITY_SYNONYMS[word]

    for city in KNOWN_CITIES:
        if re.search(rf"\b{re.escape(city.lower())}\b", query_clean.lower()):
            return "Bangalore" if city.lower() == "bengaluru" else city

    # 2. Common preposition patterns: "in <City>", "for <City>", "at <City>", "of <City>"
    match = re.search(
        r"\b(?:in|at|for|of|around|near)\s+([A-Za-z\s]+?)(?:\?|\.|\,|$|\b(?:today|tomorrow|now|currently|this|morning|evening|night)\b)",
        query_clean,
        re.IGNORECASE
    )
    if match:
        candidate = match.group(1).strip()
        candidate = re.sub(r"\b(the|a|an|current|weather|temperature|city)\b", "", candidate, flags=re.IGNORECASE).strip()
        if len(candidate) > 1:
            return normalize_city_name(candidate)

    # 3. Fallback: filter common question words and check remaining words
    stop_words = {
        "what", "is", "the", "weather", "temperature", "forecast", "how", "hot", "cold",
        "rain", "raining", "rainy", "umbrella", "jacket", "coat", "should", "i", "carry",
        "wear", "in", "for", "at", "today", "now", "currently", "degrees", "celsius"
    }
    words = [w for w in query_words if w not in stop_words]
    if words:
        candidate_phrase = " ".join(words)
        return normalize_city_name(candidate_phrase)

    return "Chennai"


def geocode_location(location: str) -> Optional[Dict[str, Any]]:
    """
    Geocodes city name into lat/long via Open-Meteo Geocoding API.
    """
    normalized_loc = normalize_city_name(location)
    try:
        params = {
            "name": normalized_loc.strip(),
            "count": 1,
            "language": "en",
            "format": "json"
        }
        response = requests.get(GEOCODING_URL, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if not results:
            # Fallback retry with raw query if normalized differed
            if normalized_loc != location:
                params["name"] = location.strip()
                response = requests.get(GEOCODING_URL, params=params, timeout=8)
                data = response.json()
                results = data.get("results", [])
            if not results:
                return None
        result = results[0]
        return {
            "name": result.get("name"),
            "country": result.get("country", ""),
            "admin1": result.get("admin1", ""),
            "latitude": result.get("latitude"),
            "longitude": result.get("longitude"),
            "timezone": result.get("timezone", "UTC")
        }
    except Exception:
        return None


def get_weather(location: str = "Chennai") -> Dict[str, Any]:
    """
    Fetches real-time weather and hourly forecast from Open-Meteo API.
    """
    place = geocode_location(location)
    if not place:
        # Fallback to Chennai if geocode fails
        place = {
            "name": location,
            "country": "India",
            "admin1": "Tamil Nadu",
            "latitude": 13.0827,
            "longitude": 80.2707,
            "timezone": "Asia/Kolkata"
        }

    try:
        params = {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "precipitation",
                "weather_code",
                "wind_speed_10m",
                "surface_pressure"
            ],
            "hourly": [
                "temperature_2m",
                "precipitation_probability",
                "relative_humidity_2m",
                "weather_code"
            ],
            "timezone": "auto",
            "forecast_days": 1
        }
        response = requests.get(FORECAST_URL, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()

        current = data.get("current", {})
        code = current.get("weather_code", 0)
        condition = WEATHER_CODES.get(code, "Clear sky")
        icon = WEATHER_ICONS.get(code, "🌤️")

        return {
            "success": True,
            "tool": "weather",
            "location": place["name"],
            "country": place.get("country", ""),
            "admin1": place.get("admin1", ""),
            "coordinates": f"{place['latitude']:.2f}°N, {place['longitude']:.2f}°E",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": data.get("timezone", place["timezone"]),
            "data": {
                "temperature": current.get("temperature_2m", 0.0),
                "feels_like": current.get("apparent_temperature", 0.0),
                "humidity": current.get("relative_humidity_2m", 0),
                "precipitation": current.get("precipitation", 0.0),
                "wind_speed": current.get("wind_speed_10m", 0.0),
                "pressure": current.get("surface_pressure", 1013.25),
                "condition": condition,
                "icon": icon,
                "weather_code": code
            },
            "hourly": data.get("hourly", {})
        }
    except requests.RequestException as e:
        return {
            "success": False,
            "tool": "weather",
            "location": location,
            "message": f"Could not connect to Open-Meteo service: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "tool": "weather",
            "location": location,
            "message": f"Weather processing error: {str(e)}"
        }


def weather_tool(query: str) -> Dict[str, Any]:
    """Main tool entry point registered with ToolRouter."""
    loc = extract_location(query)
    return get_weather(loc)