from langchain_core.tools import tool
from typing import Optional, List, Dict, Any
from app.services.mocks import (
    mock_flight_search,
    mock_hotel_search,
    mock_weather_forecast,
    mock_maps_route,
    mock_currency_conversion,
    mock_restaurant_search,
    mock_policy_document,
    mock_flight_booking,
    mock_hotel_booking,
    mock_charge_payment
)

@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """
    Search for flights between origin and destination on a specific date.
    Args:
        origin: The starting airport code or city name (e.g. NYC, London).
        destination: The destination airport code or city name (e.g. PAR, Paris).
        date: The date of the flight in YYYY-MM-DD format.
    Returns:
        A list of available flights with details like airline, flight number, times, and price.
    """
    flights = mock_flight_search(origin, destination, date)
    if not flights:
        return f"No flights found from {origin} to {destination} on {date}."
    
    result = f"Available flights from {origin} to {destination} on {date}:\n"
    for f in flights:
        result += f"- {f['airline']} {f['flight_number']}: Departs {f['departure']}, Arrives {f['arrival']}, Price: ${f['price_usd']} (Stops: {f['stops']})\n"
    return result

@tool
def search_hotels(city: str, checkin_date: str, checkout_date: str, budget_category: Optional[str] = None) -> str:
    """
    Search for hotels in a city with check-in and check-out dates and an optional budget category.
    Args:
        city: The city name (e.g. Paris, Tokyo).
        checkin_date: The check-in date in YYYY-MM-DD format.
        checkout_date: The check-out date in YYYY-MM-DD format.
        budget_category: Optional budget preference ('budget', 'mid-range', 'luxury').
    Returns:
        A list of matching hotels with their ratings, prices, and amenities.
    """
    hotels = mock_hotel_search(city, checkin_date, checkout_date, budget_category)
    if not hotels:
        return f"No hotels found in {city}."
    
    result = f"Recommended hotels in {city} ({checkin_date} to {checkout_date}):\n"
    for h in hotels:
        result += f"- {h['name']} ({h['category'].title()}): Rating {h['rating']}, ${h['price_per_night_usd']}/night, Amenities: {', '.join(h['amenities'])}\n"
    return result

@tool
def get_weather(city: str, date: str) -> str:
    """
    Get the weather forecast for a specific city on a given date.
    Args:
        city: The name of the city (e.g. Tokyo, Rome).
        date: The date in YYYY-MM-DD format.
    Returns:
        Weather forecast details including temperature, conditions, and wind speed.
    """
    forecast = mock_weather_forecast(city, date)
    return (
        f"Weather for {forecast['city']} on {forecast['date']}:\n"
        f"- Temperature: {forecast['temperature_celsius']}°C\n"
        f"- Condition: {forecast['condition']}\n"
        f"- Humidity: {forecast['humidity_percentage']}%\n"
        f"- Wind Speed: {forecast['wind_speed_kph']} kph"
    )

@tool
def get_route_and_distance(origin: str, destination: str, travel_mode: str = "driving") -> str:
    """
    Get route distance, estimated travel time, and travel cost between two points.
    Args:
        origin: The start location.
        destination: The end location.
        travel_mode: The mode of transport ('driving' or 'flying'). Defaults to 'driving'.
    Returns:
        Details on distance, travel time, and estimated cost.
    """
    route = mock_maps_route(origin, destination, travel_mode)
    return (
        f"Route information from {route['origin']} to {route['destination']} via {route['mode']}:\n"
        f"- Distance: {route['distance_km']} km\n"
        f"- Duration: {route['duration_hours']} hours\n"
        f"- Estimated Travel Cost: ${route['estimated_cost_usd']}"
    )

@tool
def convert_currency(from_currency: str, to_currency: str, amount: float = 1.0) -> str:
    """
    Convert an amount from one currency to another using current exchange rates.
    Args:
        from_currency: The currency code to convert from (e.g. USD, EUR).
        to_currency: The currency code to convert to (e.g. INR, JPY).
        amount: The amount of money to convert. Defaults to 1.0.
    Returns:
        The conversion details and exchange rate.
    """
    conv = mock_currency_conversion(from_currency, to_currency, amount)
    return (
        f"Currency Conversion:\n"
        f"- {conv['amount']} {conv['from']} = {conv['converted_amount']} {conv['to']}\n"
        f"- Exchange Rate: {conv['exchange_rate']}"
    )

@tool
def search_restaurants(city: str, cuisine: Optional[str] = None) -> str:
    """
    Search for restaurants in a city, optionally filtered by cuisine.
    Args:
        city: The city name (e.g. Paris, New York).
        cuisine: Optional cuisine type (e.g. 'Italian', 'Indian', 'Japanese').
    Returns:
        A list of recommended restaurants with ratings, average prices, and popular dishes.
    """
    restaurants = mock_restaurant_search(city, cuisine)
    if not restaurants:
        return f"No restaurants matching the criteria found in {city}."
    
    result = f"Recommended restaurants in {city}:\n"
    for r in restaurants:
        result += f"- {r['name']}: {r['cuisine']} cuisine, Rating {r['rating']}, Avg Price: ${r['average_price_usd']}, Popular Dish: {r['popular_dish']}\n"
    return result

@tool
def get_visa_information(nationality: str, destination_country: str) -> str:
    """
    Get visa requirements and information for a traveler of a specific nationality visiting a destination country.
    Args:
        nationality: The nationality/citizenship of the traveler (e.g. Indian, American).
        destination_country: The country they plan to visit (e.g. France, Japan).
    Returns:
        Visa requirements, typical visa validity, and processing notes.
    """
    nationality = nationality.title().strip()
    destination_country = destination_country.title().strip()
    
    # Simple logic to determine mock visa info
    if nationality == destination_country:
        return f"No visa required. You are traveling within your home country: {destination_country}."
        
    # Standard responses
    visa_types = ["Visa Free / Visa on Arrival", "eVisa Required", "Consular Visa Required"]
    hash_val = sum(ord(c) for c in nationality + destination_country) % 3
    visa_type = visa_types[hash_val]
    
    notes = {
        "Visa Free / Visa on Arrival": "Valid for up to 90 days of tourism. Ensure passport has at least 6 months validity.",
        "eVisa Required": "Must apply online at least 72 hours before departure. Typically processed in 3 business days. Cost: ~$50.",
        "Consular Visa Required": "Must schedule an appointment at the nearest embassy/consulate. Requires travel insurance, bank statements, and flight itinerary."
    }
    
    return (
        f"Visa Information for {nationality} passport holders visiting {destination_country}:\n"
        f"- Requirement Status: {visa_type}\n"
        f"- Key Condition: {notes[visa_type]}\n"
        f"- Recommended action: Always verify double-entry requirements if traveling to multiple countries."
    )

@tool
def get_policy_document(policy_type: str) -> str:
    """
    Retrieve travel policy document rules (e.g. cancellation, refund, fare change policy).
    Args:
        policy_type: The type of policy to query (e.g., 'refund', 'fare change', 'cancellation').
    Returns:
        The text representation of the policy document rules.
    """
    return mock_policy_document(policy_type)

@tool
def book_flight(origin: str, destination: str, date: str, price_usd: float) -> str:
    """
    Book a flight ticket.
    Args:
        origin: The departure city/airport.
        destination: The destination city/airport.
        date: Flight date.
        price_usd: Flight price.
    Returns:
        A confirmation status of the booked flight.
    """
    return mock_flight_booking(origin, destination, date, price_usd)

@tool
def book_hotel(hotel_name: str, checkin_date: str, checkout_date: str, price_usd: float) -> str:
    """
    Book a hotel room.
    Args:
        hotel_name: The name of the hotel to book.
        checkin_date: Check-in date.
        checkout_date: Check-out date.
        price_usd: Nightly price.
    Returns:
        A confirmation status of the hotel booking.
    """
    return mock_hotel_booking(hotel_name, checkin_date, checkout_date, price_usd)

@tool
def charge_payment(amount_usd: float, payment_method: str = "credit card") -> str:
    """
    Process payment charging for a booking.
    Args:
        amount_usd: The amount to charge.
        payment_method: The payment method type. Defaults to 'credit card'.
    Returns:
        The charge processing response/status.
    """
    return mock_charge_payment(amount_usd, payment_method)

# List of all tools
travel_tools = [
    search_flights,
    search_hotels,
    get_weather,
    get_route_and_distance,
    convert_currency,
    search_restaurants,
    get_visa_information,
    get_policy_document,
    book_flight,
    book_hotel,
    charge_payment
]

# Map tool names to actual functions for execution
tools_map = {tool.name: tool for tool in travel_tools}

