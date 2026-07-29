from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

def mock_flight_search(origin: str, destination: str, date: str) -> List[Dict[str, Any]]:
    """Mock flight search API."""
    # Normalize inputs
    origin = origin.upper().strip()
    destination = destination.upper().strip()
    
    # Generate some realistic flights
    return [
        {
            "flight_number": "TX-101",
            "airline": "SkyFlow Airlines",
            "origin": origin,
            "destination": destination,
            "departure": f"{date}T08:00:00",
            "arrival": f"{date}T10:30:00",
            "price_usd": 250.0,
            "stops": 0
        },
        {
            "flight_number": "TX-202",
            "airline": "Global Connect",
            "origin": origin,
            "destination": destination,
            "departure": f"{date}T14:15:00",
            "arrival": f"{date}T17:45:00",
            "price_usd": 185.0,
            "stops": 1
        },
        {
            "flight_number": "TX-303",
            "airline": "StarAirways",
            "origin": origin,
            "destination": destination,
            "departure": f"{date}T19:30:00",
            "arrival": f"{date}T22:00:00",
            "price_usd": 310.0,
            "stops": 0
        }
    ]

def mock_hotel_search(city: str, checkin: str, checkout: str, budget_category: Optional[str] = None) -> List[Dict[str, Any]]:
    """Mock hotel search API."""
    city = city.title().strip()
    
    hotels = [
        {
            "name": "Grand Palace Hotel",
            "city": city,
            "rating": 4.8,
            "price_per_night_usd": 220.0,
            "amenities": ["Wi-Fi", "Pool", "Gym", "Breakfast Included"],
            "category": "luxury"
        },
        {
            "name": "Cozy Stay Inn",
            "city": city,
            "rating": 4.2,
            "price_per_night_usd": 95.0,
            "amenities": ["Wi-Fi", "Free Parking"],
            "category": "budget"
        },
        {
            "name": "Urban Central Suites",
            "city": city,
            "rating": 4.5,
            "price_per_night_usd": 150.0,
            "amenities": ["Wi-Fi", "Kitchenette", "Gym"],
            "category": "mid-range"
        }
    ]
    
    if budget_category:
        hotels = [h for h in hotels if h["category"] == budget_category.lower()]
    return hotels

def mock_weather_forecast(city: str, date: str) -> Dict[str, Any]:
    """Mock weather API."""
    city = city.title().strip()
    # Create simple hash from city name to vary temperature
    temp_offset = sum(ord(c) for c in city) % 15
    base_temp = 20 + temp_offset
    
    return {
        "city": city,
        "date": date,
        "temperature_celsius": base_temp,
        "condition": "Partly Cloudy" if temp_offset % 2 == 0 else "Sunny",
        "humidity_percentage": 55 + (temp_offset % 5) * 5,
        "wind_speed_kph": 12 + temp_offset % 3
    }

def mock_maps_route(origin: str, destination: str, mode: str = "driving") -> Dict[str, Any]:
    """Mock maps/routing API."""
    # Distance estimation based on names
    char_diff = abs(len(origin) - len(destination)) + 1
    distance_km = char_diff * 45 + 10
    duration_hours = distance_km / 80 if mode == "driving" else distance_km / 700
    
    return {
        "origin": origin,
        "destination": destination,
        "mode": mode,
        "distance_km": round(distance_km, 1),
        "duration_hours": round(duration_hours, 1),
        "estimated_cost_usd": round(distance_km * 0.15, 2) if mode == "driving" else round(100 + distance_km * 0.08, 2)
    }

def mock_currency_conversion(from_currency: str, to_currency: str, amount: float = 1.0) -> Dict[str, Any]:
    """Mock currency conversion API."""
    from_currency = from_currency.upper().strip()
    to_currency = to_currency.upper().strip()
    
    # Mock rates relative to USD
    rates_to_usd = {
        "USD": 1.0,
        "EUR": 0.92,
        "GBP": 0.78,
        "INR": 83.5,
        "JPY": 155.0,
        "AUD": 1.5,
        "CAD": 1.36,
    }
    
    # Fallback to dynamic generation if rate is not defined
    rate_from = rates_to_usd.get(from_currency, 1.0)
    rate_to = rates_to_usd.get(to_currency, 1.0 + (sum(ord(c) for c in to_currency) % 10) / 10.0)
    
    # Exchange rate = rate_to / rate_from
    rate = rate_to / rate_from
    converted_amount = amount * rate
    
    return {
        "from": from_currency,
        "to": to_currency,
        "amount": amount,
        "exchange_rate": round(rate, 4),
        "converted_amount": round(converted_amount, 2)
    }

def mock_restaurant_search(city: str, cuisine: Optional[str] = None) -> List[Dict[str, Any]]:
    """Mock restaurant search API."""
    city = city.title().strip()
    
    restaurants = [
        {
            "name": f"The Golden Spoon ({city})",
            "cuisine": "Italian",
            "rating": 4.7,
            "average_price_usd": 45.0,
            "popular_dish": "Truffle Tagliatelle"
        },
        {
            "name": f"Spice & Sprout ({city})",
            "cuisine": "Indian",
            "rating": 4.6,
            "average_price_usd": 25.0,
            "popular_dish": "Butter Chicken / Paneer Tikka"
        },
        {
            "name": f"Bayside Bistro ({city})",
            "cuisine": "Seafood",
            "rating": 4.4,
            "average_price_usd": 35.0,
            "popular_dish": "Grilled Sea Bass"
        },
        {
            "name": f"Sakura Sushi ({city})",
            "cuisine": "Japanese",
            "rating": 4.9,
            "average_price_usd": 60.0,
            "popular_dish": "Omakase Chef's Special"
        }
    ]
    
    if cuisine:
        cuisine_lower = cuisine.lower().strip()
        restaurants = [r for r in restaurants if cuisine_lower in r["cuisine"].lower()]
        
    return restaurants

def mock_policy_document(policy_type: str) -> str:
    """Mock policy document retrieval."""
    p_type = policy_type.lower().strip()
    if "refund" in p_type:
        return "Refund Policy: Refunds are allowed within 24 hours of booking, subject to a $50 administrative fee."
    elif "change" in p_type or "fare" in p_type:
        return "Fare Change Rules: Changes to tickets can be made up to 48 hours before departure. Basic economy tickets are non-changeable."
    elif "cancel" in p_type:
        return "Cancellation Policy: Cancellations made 7 days prior to departure receive a 100% refund. Cancellations between 7 days and 24 hours receive a 50% refund."
    else:
        return f"Standard travel policy details for {policy_type}."

def mock_flight_booking(origin: str, destination: str, date: str, price_usd: float) -> str:
    """Mock flight booking service."""
    return f"Flight successfully booked from {origin} to {destination} on {date} for ${price_usd}."

def mock_hotel_booking(hotel_name: str, checkin: str, checkout: str, price_usd: float) -> str:
    """Mock hotel booking service."""
    return f"Hotel '{hotel_name}' successfully booked checking in on {checkin} and out on {checkout} for ${price_usd}."

def mock_charge_payment(amount_usd: float, method: str) -> str:
    """Mock payment charging service."""
    return f"Successfully charged ${amount_usd} via {method}."

