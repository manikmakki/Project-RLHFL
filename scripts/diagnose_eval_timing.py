"""Check whether non-golden evaluations happened before or after the last training run.

Run inside the training container:
  docker compose exec rlhfl python3 /app/scripts/diagnose_eval_timing.py
"""
import sys
sys.path.insert(0, "/app")

from shared.config import load_config
from api.es_memory_manager import ElasticsearchMemoryManager

cfg = load_config()
mgr = ElasticsearchMemoryManager(cfg)
idx = mgr.index

# When did the last training run start?
state = mgr._load_training_state_full()
last_training = state.get("last_training")
print(f"Last training timestamp: {last_training}")
print()

if not last_training:
    print("No training timestamp recorded — cannot check timing.")
    sys.exit(0)

training_ts = last_training.isoformat()

def count(query):
    return mgr.client.count(index=idx, query=query)["count"]

base = [
    {"term": {"data.type": "conversation"}},
    {"exists": {"field": "meta.rlhfl_evaluated_at"}},
    {"term": {"meta.rlhfl_is_golden": False}},
    {"term": {"meta.rlhfl_is_noise": False}},
]

evaluated_before = count({"bool": {"filter": base + [
    {"range": {"meta.rlhfl_evaluated_at": {"lt": training_ts}}},
]}})

evaluated_after = count({"bool": {"filter": base + [
    {"range": {"meta.rlhfl_evaluated_at": {"gte": training_ts}}},
]}})

total_non_golden = count({"bool": {"filter": base}})

print(f"Non-golden evaluated BEFORE training ({training_ts[:19]}):  {evaluated_before}")
print(f"Non-golden evaluated AFTER  training:                          {evaluated_after}")
print(f"Total non-golden evaluated now:                                {total_non_golden}")
print()

if evaluated_before == 0 and evaluated_after > 0:
    print("DIAGNOSIS: all non-golden evaluations happened AFTER the training run.")
    print("  At training time only the 12 golden examples were evaluated.")
    print("  The next scheduled training cycle will pick up all 120 SFT-usable samples.")
elif evaluated_before > 0:
    print(f"DIAGNOSIS: {evaluated_before} non-golden were evaluated before training — those should have been used.")
    print("  Check the 'Retrieved X top-weighted interactions' log line from that run.")
