# mcp_flight_server/tools/toolhandler.py
import os, requests, json, time
from typing import Dict, Any, List
from pydantic import BaseModel, Field

# --- Autenticación (sin cambios) ---
ACCESS_TOKEN = None
TOKEN_EXPIRATION = 0

def get_access_token(retries: int = 2) -> str | None:
    global ACCESS_TOKEN, TOKEN_EXPIRATION
    if ACCESS_TOKEN and time.time() < TOKEN_EXPIRATION:
        print("[AmadeusAuth] ♻️ Reutilizando token de vuelo cacheado.")
        return ACCESS_TOKEN
    AMADEUS_BASE_URL = os.getenv("AMADEUS_BASE_URL", "https://test.api.amadeus.com")
    AMADEUS_API_KEY = os.getenv("AMADEUS_API_KEY", "").strip()
    AMADEUS_API_SECRET = os.getenv("AMADEUS_API_SECRET", "").strip()
    if not AMADEUS_API_KEY or not AMADEUS_API_SECRET: return None
    auth_url = f"{AMADEUS_BASE_URL}/v1/security/oauth2/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    payload = f"grant_type=client_credentials&client_id={AMADEUS_API_KEY}&client_secret={AMADEUS_API_SECRET}"
    for _ in range(retries):
        try:
            resp = requests.post(auth_url, data=payload, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                ACCESS_TOKEN = data.get("access_token")
                expires_in = data.get("expires_in", 1799)
                TOKEN_EXPIRATION = time.time() + expires_in - 30
                print(f"[AmadeusAuth] ✅ Token de vuelo OK (expira en {expires_in}s)")
                return ACCESS_TOKEN
        except Exception as e: print(f"[AmadeusAuth] Excepción: {e}")
    return None

# --- ✅ Modelo de Argumentos Actualizado ---
class FlightSearchArgs(BaseModel):
    origin: str = Field(..., description="IATA de origen (ej. MAD)")
    destination: str = Field(..., description="IATA de destino (ej. MRS)")
    date: str = Field(..., description="Fecha de salida (YYYY-MM-DD)")
    adults: int = Field(1, description="Número de pasajeros adultos")
    max_results: int = 5

from datetime import datetime, timedelta

def normalize_date(date_str: str) -> str:
    """Ajusta la fecha para que nunca sea anterior a hoy."""
    try:
        # Convertimos la cadena a objeto fecha
        user_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        # Si el formato no es válido, asignamos mañana
        return (datetime.utcnow().date() + timedelta(days=1)).isoformat()

    today = datetime.utcnow().date()
    if user_date <= today:
        # Si la fecha es pasada o hoy mismo, la movemos a mañana
        user_date = today + timedelta(days=1)

    return user_date.isoformat()


# --- Lógica de Búsqueda Enriquecida ---
def search_flights(args: FlightSearchArgs) -> Dict[str, Any]:
    token = get_access_token()
    if not token:
        return {"flights": [], "error": "No se pudo obtener token de Amadeus"}

    AMADEUS_BASE_URL = os.getenv("AMADEUS_BASE_URL", "https://test.api.amadeus.com")
    url = f"{AMADEUS_BASE_URL}/v2/shopping/flight-offers"
    headers = {"Authorization": f"Bearer {token}"}
    
    # ✅ Se añade el parámetro 'adults' a la llamada
    params = {
        "originLocationCode": args.origin,
        "destinationLocationCode": args.destination,
        "departureDate": normalize_date(args.date),
        "adults": args.adults,
        "max": args.max_results,
        "currencyCode": "EUR"
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        if response.status_code != 200:
            print(f"[FlightTool] Error API {response.status_code}: {response.text}")
            return {"flights": [], "error": f"API de Amadeus respondió con error {response.status_code}: {response.text}"}
        
        data = response.json()
        flights = []
        carrier_codes = data.get("dictionaries", {}).get("carriers", {})

        for offer in data.get("data", []):
            itinerary = offer.get("itineraries", [{}])[0]
            segment = itinerary.get("segments", [{}])[0]
            carrier_code = segment.get("carrierCode")
            airline_name = carrier_codes.get(carrier_code, carrier_code)

            flights.append({
                "airline": airline_name,
                "flight_number": f"{segment.get('carrierCode')} {segment.get('number')}",
                "origin": segment.get("departure", {}).get("iataCode"),
                "destination": segment.get("arrival", {}).get("iataCode"),
                "departure_time": segment.get("departure", {}).get("at"),
                "arrival_time": segment.get("arrival", {}).get("at"),
                "duration": itinerary.get("duration", "").replace("PT", "").replace("H", "h ").replace("M", "m"),
                "stops": len(itinerary.get("segments", [])) - 1,
                "price": float(offer.get("price", {}).get("total", 0.0)),
                "currency": offer.get("price", {}).get("currency", "EUR"),
            })
        print(f"[FlightTool] Encontrados {len(flights)} vuelos de {args.origin} a {args.destination} para {args.adults} adulto(s)")
        return {"flights": flights}
    except Exception as e:
        print(f"[FlightTool] Excepción en la búsqueda: {e}")
        return {"flights": [], "error": str(e)}

# --- REGISTRO Y DISPATCHER ---
def list_tools():
    return [{"name": "flight.search_flights", "description": "Busca vuelos directos para un número de adultos.", "parameters": FlightSearchArgs.schema()}]

def call_tool(name: str, args: dict):
    if name == "flight.search_flights":
        try:
            parsed_args = FlightSearchArgs(**args)
            return search_flights(parsed_args)
        except Exception as e:
            return {"flights": [], "error": f"Argumentos inválidos: {e}"}
    return {"error": f"Herramienta desconocida: {name}"}
