# Dream Mode

Make your AI agent "dream" by allowing hallucination, research, and exploration for a period of time, and keeping notes of what was thought of or discovered.

## Three Functions

| Function | Description |
|---|---|
| `dream(topic)` | Starts a new dream session — runs multiple LLM cycles with a high-temperature, creativity-encouraging system prompt, then saves the full journal as an Open WebUI note |
| `recall_dreams(query)` | Searches previous dream notes via the notes search API |
| `continue_dream(note_id)` | Loads an existing dream note, picks up the "seeds" from where it left off, runs more cycles, and appends to the same note |

## How It Works

- **Dream cycles** — Each session runs `DREAM_CYCLES` (default 3) sequential LLM calls. The system prompt explicitly encourages hallucination, speculation, cross-domain connections, and "what if" thinking.

- **Seed propagation** — Each cycle ends with a "Seeds for Next Cycle" section. The next cycle picks up those seeds as its starting prompt, so ideas compound and deepen across cycles.

- **Prior dream awareness** — Before the first cycle, it searches existing notes for related topics so the model can build on or challenge earlier explorations.

- **Notes API integration** — Uses `POST /api/v1/notes/create` and `POST /api/v1/notes/{id}/update` with Bearer token auth to persist the dream journal as a proper Open WebUI note.

## Configuration (Valves)

- `OPENWEBUI_API_URL` / `OPENWEBUI_API_KEY` — required for notes persistence
- `LLM_API_URL` / `LLM_MODEL` — the model that does the dreaming
- `DREAM_CYCLES` — how many rounds (default 3)
- `DREAM_TEMPERATURE` — creativity dial (default 1.3)
- `DREAM_MAX_TOKENS` — per-cycle token budget (default 2048)

## Installation

Paste the file contents into Open WebUI under **Workspace > Tools > Create**, or import the `.py` file directly.
