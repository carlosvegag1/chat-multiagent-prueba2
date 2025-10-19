# 🧭 Chat Multi-Agent Travel

Sistema inteligente de planificación de viajes con arquitectura multiagente.  
Integrado con FastAPI, Next.js, Whisper (OpenAI) y micro-servicios MCP (Model Context Protocol).

---

## 🧠 Descripción general

Este proyecto es una demostración práctica de un sistema multiagente diseñado para generar planes de viaje completos a partir de lenguaje natural.

El usuario puede simplemente escribir algo como:

> "Planea un viaje de 4 días a Manchester con mi novia."

Y el sistema:

- Interpreta el mensaje con un modelo de lenguaje (LLM)
- Coordina varios agentes especializados (vuelos, hoteles, destinos y cálculo)
- Fusiona la información y genera un itinerario completo de 72 h, con presupuesto, lugares recomendados y resumen narrativo
- Permite enviar mensajes de texto o de voz, utilizando Whisper para la transcripción automática

---

## 🧩 Arquitectura del sistema

| Componente | Tecnología | Descripción |
|-----------|-----------|-----------|
| 🧠 Backend | FastAPI + Whisper | Orquesta la lógica multiagente, gestiona usuarios, conversaciones y transcripción de audio |
| 💬 Frontend | Next.js (React + Tailwind) | Interfaz de chat conversacional, conexión con la API y renderización de itinerarios estructurados |
| ✈️ Agente de vuelos | MCP (Python + FastAPI) | Simula consultas a la API de Amadeus para buscar vuelos |
| 🏨 Agente de hoteles | MCP (Python + FastAPI) | Gestiona sugerencias de hoteles |
| 🌍 Agente de destinos | MCP (Python + FastAPI) | Genera planes y puntos de interés de la ciudad |
| 🧮 Agente de cálculo | MCP (Python + FastAPI) | Calcula presupuestos y agrega resultados |

Todos los servicios se ejecutan y comunican dentro de Docker, utilizando una red interna (`multiagent_net`) que garantiza aislamiento y rendimiento.

---

## ⚙️ 1) Requisitos

✅ **Docker Desktop** (Windows / macOS / Linux)

✅ **Conexión a internet** para las dependencias iniciales

✅ **(Opcional) Claves reales para APIs externas:**
- `OPENAI_API_KEY`
- `AMADEUS_API_KEY`
- `AMADEUS_API_SECRET`
- `WEATHER_API_KEY`

---

## 🔐 2) Configuración de variables de entorno

Copia el archivo de ejemplo y añade tus claves (si las tienes):

```bash
cp .env.example .env
```

Edita el nuevo `.env`:

```bash
OPENAI_API_KEY=sk-xxxx
AMADEUS_API_KEY=xxxx
AMADEUS_API_SECRET=xxxx
WEATHER_API_KEY=xxxx
```

Si no añades claves, el sistema seguirá funcionando en modo simulado, generando datos de ejemplo.

---

## 🚀 3) Arranque rápido (con Docker)

En la raíz del proyecto:

```bash
docker compose up -d --build
```

Esto levantará todos los servicios de forma automática.

| Servicio | Puerto | Descripción |
|----------|--------|-----------|
| 🌍 Frontend | [http://127.0.0.1:3000](http://127.0.0.1:3000) | Interfaz de chat |
| ⚙️ Backend | [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) | API y documentación Swagger |
| 🧮 Calc Agent | [http://127.0.0.1:8770/docs](http://127.0.0.1:8770/docs) | Servicio de cálculos |
| ✈️ Flights Agent | [http://127.0.0.1:8771/docs](http://127.0.0.1:8771/docs) | Servicio de vuelos |
| 🏨 Hotels Agent | [http://127.0.0.1:8772/docs](http://127.0.0.1:8772/docs) | Servicio de hoteles |
| 🌍 Destinations Agent | [http://127.0.0.1:8773/docs](http://127.0.0.1:8773/docs) | Servicio de destinos |

---

## 🧠 4) Verificar el estado del sistema

Comprobar que todos los servicios estén activos:

```bash
docker compose ps
```

Si aparecen con `Up (healthy)`, el sistema está funcionando. También puedes verificar cada servicio:

```bash
curl -I http://127.0.0.1:8000/docs
curl -I http://127.0.0.1:8770/docs
curl -I http://127.0.0.1:8771/docs
curl -I http://127.0.0.1:8772/docs
curl -I http://127.0.0.1:8773/docs
```

---

## 💬 5) Uso del chat

Abre el navegador y entra a: **http://127.0.0.1:3000**

Verás una interfaz conversacional limpia y moderna. Puedes escribir mensajes naturales como:

- "Planea un viaje a París de 3 días con mi pareja."
- "Qué información tienes sobre Londres."
- "Planea un viaje de negocios a Berlín."

El sistema recuerda tus viajes anteriores gracias a su persistencia JSON (`backend/data/v2/users/`).

También puedes grabar mensajes de voz con el botón de micrófono; el sistema utiliza Whisper para transcribir tu audio automáticamente y responder.

---

## 🔊 6) Pruebas por API (modo avanzado)

Puedes comunicarte directamente con la API del backend desde PowerShell o terminal:

```powershell
$body = @{
  user     = "Demo"
  message  = "Planea un viaje de 2 días a Palma de Mallorca"
  convo_id = ""
} | ConvertTo-Json

curl.exe -s -X POST "http://127.0.0.1:8000/chat/" `
  -H "Content-Type: application/json" `
  --data-raw $body
```

Respuesta esperada:

```json
{
  "conversation_id": "20251017T223205_652666",
  "intent": "PLAN_TRIP",
  "reply_text": "Aquí tienes tu plan para Palma de Mallorca...",
  "structured_data": {...},
  "agents_called": ["FlightAgent", "HotelAgent", "DestinationAgent"]
}
```

---

## 🪵 7) Monitorizar y depurar

Ver los logs en tiempo real:

```bash
docker compose logs -f backend
```

Ver todos los servicios:

```bash
docker compose logs --tail=200
```

Buscar errores:

```bash
docker compose logs backend | findstr /I "ERROR"
```

---

## 🧰 8) Solución de problemas frecuentes

**❌ Backend llama a `127.0.0.1` dentro de Docker**

Los agentes ya están configurados como `flights:8771`, `destinations:8773`, etc.  
Comprueba en los logs que veas:

```
FlightAgent conectado a: http://flights:8771
DestinationAgent conectado a: http://destinations:8773
```

**⚠️ Warning: `WEATHER_API_KEY` no definida**

Es opcional. Si quieres quitar el aviso, añade la variable en tu `.env`.

**🎙️ Whisper no transcribe audio**

Asegúrate de que `ffmpeg` está instalado (el Dockerfile ya lo hace).  
Si ejecutas fuera de Docker, instálalo manualmente:

```bash
sudo apt install ffmpeg
```

---

## 💻 9) Ejecución sin Docker (modo desarrollador)

```bash
# MCPs
cd mcp_flight_server && uvicorn server:app --port 8771
cd mcp_hotel_server && uvicorn server:app --port 8772
cd mcp_destination_server && uvicorn server:app --port 8773
cd mcp_calc_server && uvicorn server:app --port 8770

# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend
cd frontend
npm install
npm run dev
```

---

## 📦 10) Estructura del proyecto

```
chat-multiagent/
├── backend/
│   ├── main.py
│   ├── app/core/orchestrator/
│   ├── data/v2/users/
│   └── requirements.txt
├── frontend/
│   ├── components/Chat.tsx
│   ├── app/page.tsx
│   ├── app/layout.tsx
│   ├── public/icons/
│   └── styles/globals.css
├── mcp_calc_server/
├── mcp_flight_server/
├── mcp_hotel_server/
├── mcp_destination_server/
├── Dockerfile.backend
├── Dockerfile.frontend
├── Dockerfile.mcp
├── docker-compose.yml
└── README.md
```

---

## 🧩 11) Qué aprenderás con este proyecto

- Cómo construir una arquitectura multiagente funcional
- Cómo integrar APIs externas (OpenAI, Amadeus, Whisper)
- Cómo usar Docker Compose para coordinar múltiples servicios
- Cómo implementar un frontend conversacional con React y Next.js
- Cómo estructurar y persistir conversaciones con FastAPI
- Cómo construir microservicios MCP compatibles con el Model Context Protocol

---

## 🤝 Créditos y autoría

Proyecto educativo desarrollado en el marco de un Trabajo Fin de Máster, orientado a la exploración de sistemas multiagente, IA generativa y protocolos de comunicación entre agentes (MCP + A2A).

---

Hecho con ❤️ para que lo arranques y pruebes en minutos.
