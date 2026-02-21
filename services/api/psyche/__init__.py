"""
Project RLHFL - Psyche Module (Id / Ego / Superego)

Freudian-inspired agent architecture providing:
- Id: Reward-driven engine (immediate success signals)
- Ego: Planner & executor (fast per-request + slow background)
- Superego: Discrimination layer (real danger vs. false alarms)

Import components directly from their submodules:
    from api.psyche.orchestrator import PsycheOrchestrator
    from api.psyche.superego import Superego
    from api.psyche.ego_fast import EgoFast
    from api.psyche.id_engine import IdEngine
    from api.psyche.ego_slow import EgoSlow
"""
