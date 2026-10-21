"""
Unit tests for Open-Meteo Weather service and endpoints.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from main import app
from weather.service import geocode_district, get_weather_by_coords, WMO_CODE_MAP

client = TestClient(app)


# ---------- Service Tests ----------

def test_geocode_empty_query():
    with pytest.raises(ValueError, match="Search query cannot be empty"):
        geocode_district("")


def test_invalid_coordinates():
    with pytest.raises(ValueError, match="Latitude must be between"):
        get_weather_by_coords(95.0, 77.0)

    with pytest.raises(ValueError, match="Longitude must be between"):
        get_weather_by_coords(20.0, 190.0)


@patch("weather.service.requests.get")
def test_geocode_district_mocked(mock_get):
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "results": [
            {
                "name": "Ludhiana",
                "admin1": "Punjab",
                "country": "India",
                "latitude": 30.912,
                "longitude": 75.854,
            }
        ]
    }
    mock_get.return_value = mock_resp

    results = geocode_district("Ludhiana")
    assert len(results) == 1
    assert results[0]["name"] == "Ludhiana"
    assert results[0]["state"] == "Punjab"
    assert results[0]["latitude"] == 30.912
    assert "Ludhiana, Punjab, India" in results[0]["display_name"]


@patch("weather.service.requests.get")
def test_get_weather_by_coords_mocked(mock_get):
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "current": {
            "temperature_2m": 28.4,
            "relative_humidity_2m": 62,
            "precipitation": 0.0,
            "weather_code": 1,
        },
        "daily": {
            "precipitation_sum": [2.5, 5.0, 12.5, 0.0],
            "temperature_2m_mean": [27.0, 28.0, 26.5, 27.5],
        },
    }
    mock_get.return_value = mock_resp

    weather = get_weather_by_coords(30.912, 75.854, location_name="Ludhiana, Punjab")
    assert weather["temperature"] == 28.4
    assert weather["humidity"] == 62.0
    assert weather["rainfall"] == 20.0
    assert weather["condition"] == "Mainly Clear"
    assert weather["icon"] == "🌤️"
    assert weather["location_name"] == "Ludhiana, Punjab"


# ---------- API Endpoint Tests ----------

@patch("weather.service.requests.get")
def test_api_weather_search(mock_get):
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "results": [
            {
                "name": "Varanasi",
                "admin1": "Uttar Pradesh",
                "country": "India",
                "latitude": 25.317,
                "longitude": 83.010,
            }
        ]
    }
    mock_get.return_value = mock_resp

    res = client.get("/weather/search?query=Varanasi")
    assert res.status_code == 200
    data = res.json()
    assert data["count"] == 1
    assert data["results"][0]["name"] == "Varanasi"


def test_api_weather_search_empty():
    res = client.get("/weather/search?query=")
    assert res.status_code == 400


@patch("weather.service.requests.get")
def test_api_weather_coordinates(mock_get):
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "current": {
            "temperature_2m": 31.0,
            "relative_humidity_2m": 45,
            "precipitation": 0.0,
            "weather_code": 0,
        },
        "daily": {
            "precipitation_sum": [10.0, 15.0],
            "temperature_2m_mean": [29.5, 30.0],
        },
    }
    mock_get.return_value = mock_resp

    res = client.get("/weather/coordinates?lat=25.317&lon=83.010&location_name=Varanasi")
    assert res.status_code == 200
    data = res.json()
    assert data["temperature"] == 31.0
    assert data["humidity"] == 45.0
    assert data["rainfall"] == 25.0


def test_api_weather_coordinates_invalid():
    res = client.get("/weather/coordinates?lat=95.0&lon=83.010")
    assert res.status_code == 422


@patch("weather.service.requests.get")
def test_api_weather_district(mock_get):
    # Mocking two calls: geocoding, then forecast
    mock_geo_resp = MagicMock()
    mock_geo_resp.raise_for_status.return_value = None
    mock_geo_resp.json.return_value = {
        "results": [
            {
                "name": "Patna",
                "admin1": "Bihar",
                "country": "India",
                "latitude": 25.594,
                "longitude": 85.136,
            }
        ]
    }

    mock_forecast_resp = MagicMock()
    mock_forecast_resp.raise_for_status.return_value = None
    mock_forecast_resp.json.return_value = {
        "current": {
            "temperature_2m": 29.0,
            "relative_humidity_2m": 50,
            "precipitation": 0.0,
            "weather_code": 2,
        },
        "daily": {
            "precipitation_sum": [40.0, 20.0],
            "temperature_2m_mean": [28.0, 28.5],
        },
    }

    mock_get.side_effect = [mock_geo_resp, mock_forecast_resp]

    res = client.get("/weather/district?query=Patna")
    assert res.status_code == 200
    data = res.json()
    assert data["temperature"] == 29.0
    assert data["rainfall"] == 60.0
    assert "Patna" in data["location_name"]
