"""
Open-Meteo Weather & Geocoding Service for Kissaan AI.

Fetches real-time temperature, humidity, and cumulative precipitation
for any district, city, or GPS coordinate without requiring API keys.
"""

from typing import Any, Dict, List, Optional
import requests

GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"

# WMO Weather interpretation codes (WW)
WMO_CODE_MAP = {
    0: ("Clear Sky", "☀️"),
    1: ("Mainly Clear", "🌤️"),
    2: ("Partly Cloudy", "⛅"),
    3: ("Overcast", "☁️"),
    45: ("Foggy", "🌫️"),
    48: ("Depositing Rime Fog", "🌫️"),
    51: ("Light Drizzle", "🌦️"),
    53: ("Moderate Drizzle", "🌦️"),
    55: ("Dense Drizzle", "🌧️"),
    61: ("Slight Rain", "🌧️"),
    63: ("Moderate Rain", "🌧️"),
    65: ("Heavy Rain", "🌧️"),
    71: ("Slight Snow", "❄️"),
    73: ("Moderate Snow", "❄️"),
    75: ("Heavy Snow", "❄️"),
    80: ("Slight Rain Showers", "🌦️"),
    81: ("Moderate Rain Showers", "🌦️"),
    82: ("Violent Rain Showers", "⛈️"),
    95: ("Thunderstorm", "⛈️"),
    96: ("Thunderstorm with Slight Hail", "⛈️"),
    99: ("Thunderstorm with Heavy Hail", "⛈️"),
}


def geocode_district(query: str, count: int = 5, timeout: float = 6.0) -> List[Dict[str, Any]]:
    """
    Search for districts, towns, or cities using Open-Meteo Geocoding API.
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("Search query cannot be empty.")

    params = {
        "name": query,
        "count": max(1, min(count, 10)),
        "language": "en",
        "format": "json",
    }
    headers = {"User-Agent": "KissaanAI-SoilHealth/0.4.0"}

    try:
        response = requests.get(
            GEOCODING_API_URL,
            params=params,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        raise RuntimeError(f"Geocoding service error: {exc}") from exc

    results = []
    for item in data.get("results", []):
        name = item.get("name", "")
        state = item.get("admin1", "")
        country = item.get("country", "")
        lat = item.get("latitude")
        lon = item.get("longitude")

        parts = [p for p in [name, state, country] if p]
        display_name = ", ".join(parts)

        results.append({
            "name": name,
            "state": state,
            "country": country,
            "latitude": round(float(lat), 5) if lat is not None else None,
            "longitude": round(float(lon), 5) if lon is not None else None,
            "display_name": display_name,
        })

    return results


def get_weather_by_coords(
    latitude: float,
    longitude: float,
    location_name: Optional[str] = None,
    timeout: float = 7.0,
) -> Dict[str, Any]:
    """
    Fetch current and rolling seasonal agro-climatic conditions for given coordinates.
    Includes:
      - Current temperature (°C) & rolling mean temperature (°C)
      - Current relative humidity (%)
      - 37-day cumulative rainfall (past 30 days + 7-day forecast) in mm
      - WMO weather description & icon
    """
    if not (-90.0 <= latitude <= 90.0):
        raise ValueError("Latitude must be between -90 and 90 degrees.")
    if not (-180.0 <= longitude <= 180.0):
        raise ValueError("Longitude must be between -180 and 180 degrees.")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code",
        "daily": "precipitation_sum,temperature_2m_mean",
        "past_days": 30,
        "forecast_days": 7,
        "timezone": "auto",
    }
    headers = {"User-Agent": "KissaanAI-SoilHealth/0.4.0"}

    try:
        response = requests.get(
            FORECAST_API_URL,
            params=params,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        raise RuntimeError(f"Open-Meteo forecast error: {exc}") from exc

    current = data.get("current", {})
    daily = data.get("daily", {})

    temp_now = current.get("temperature_2m")
    humidity_now = current.get("relative_humidity_2m")
    weather_code = current.get("weather_code", 0)

    # Calculate 37-day precipitation sum and mean temperature
    precip_list = [p for p in daily.get("precipitation_sum", []) if p is not None]
    total_rainfall = round(sum(precip_list), 1) if precip_list else 0.0

    mean_temps = [t for t in daily.get("temperature_2m_mean", []) if t is not None]
    mean_temp = round(sum(mean_temps) / len(mean_temps), 1) if mean_temps else temp_now

    # Look up weather description and icon
    condition, icon = WMO_CODE_MAP.get(weather_code, ("Fair Weather", "🌤️"))

    return {
        "location_name": location_name or f"Coordinates ({latitude:.3f}, {longitude:.3f})",
        "latitude": round(float(latitude), 4),
        "longitude": round(float(longitude), 4),
        "temperature": round(float(temp_now), 1) if temp_now is not None else 25.0,
        "mean_temperature_30d": mean_temp,
        "humidity": round(float(humidity_now), 1) if humidity_now is not None else 65.0,
        "rainfall": total_rainfall,
        "weather_code": weather_code,
        "condition": condition,
        "icon": icon,
        "source": "Open-Meteo Weather API",
    }
