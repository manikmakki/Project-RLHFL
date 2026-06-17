"""Diagnose why training used fewer samples than the stats show.

Run inside the training container:
  docker compose exec rlhfl python3 /app/scripts/diagnose_training_data.py
"""
import sys
sys.path.insert(0, "/app")

from shared.config import load_config
from api.es_memory_manager import ElasticsearchMemoryManager
from datetime import datetime, timedelta

cfg = load_config()
mgr = ElasticsearchMemoryManager(cfg)
idx = mgr.index
max_age = cfg.elasticsearch.max_memory_age_days
cutoff = (datetime.now() - timedelta(days=max_age)).isoformat()

def count(query):
    return mgr.client.count(index=idx, query=query)["count"]

golden = count({"bool": {"filter": [
    {"term": {"data.type": "conversation"}},
    {"exists": {"field": "meta.rlhfl_evaluated_at"}},
    {"term": {"meta.rlhfl_is_golden": True}},
]}})

non_golden_all = count({"bool": {"filter": [
    {"term": {"data.type": "conversation"}},
    {"exists": {"field": "meta.rlhfl_evaluated_at"}},
    {"term": {"meta.rlhfl_is_golden": False}},
    {"term": {"meta.rlhfl_is_noise": False}},
]}})

non_golden_recent = count({"bool": {
    "filter": [
        {"term": {"data.type": "conversation"}},
        {"exists": {"field": "meta.rlhfl_evaluated_at"}},
        {"term": {"meta.rlhfl_is_golden": False}},
        {"range": {"@timestamp": {"gte": cutoff}}},
    ],
    "must_not": [{"term": {"meta.rlhfl_is_noise": True}}],
}})

noise = count({"bool": {"filter": [
    {"term": {"data.type": "conversation"}},
    {"exists": {"field": "meta.rlhfl_evaluated_at"}},
    {"term": {"meta.rlhfl_is_noise": True}},
]}})

pos_thresh, neg_thresh = 0.3, -0.3

non_golden_positive = count({"bool": {"filter": [
    {"term": {"data.type": "conversation"}},
    {"exists": {"field": "meta.rlhfl_evaluated_at"}},
    {"term": {"meta.rlhfl_is_golden": False}},
    {"term": {"meta.rlhfl_is_noise": False}},
    {"range": {"meta.rlhfl_sentiment": {"gt": pos_thresh}}},
    {"range": {"@timestamp": {"gte": cutoff}}},
]}})

non_golden_negative = count({"bool": {"filter": [
    {"term": {"data.type": "conversation"}},
    {"exists": {"field": "meta.rlhfl_evaluated_at"}},
    {"term": {"meta.rlhfl_is_golden": False}},
    {"term": {"meta.rlhfl_is_noise": False}},
    {"range": {"meta.rlhfl_sentiment": {"lt": neg_thresh}}},
    {"range": {"@timestamp": {"gte": cutoff}}},
]}})

non_golden_neutral = non_golden_recent - non_golden_positive - non_golden_negative

print(f"Age cutoff:                                    {cutoff}  ({max_age}d)")
print()
print(f"Golden examples (no age limit):                {golden:>5}   → always SFT-usable")
print(f"Non-golden, non-noise (within {max_age}d):         {non_golden_recent:>5}")
print(f"  ├─ positive  (sentiment >  {pos_thresh}):         {non_golden_positive:>5}   → SFT-usable")
print(f"  ├─ neutral   (|sentiment| ≤ {pos_thresh}):        {non_golden_neutral:>5}   → SFT-usable")
print(f"  └─ negative  (sentiment < {neg_thresh}):         {non_golden_negative:>5}   → DPO-only, dropped by SFT")
print(f"Noise (excluded entirely):                     {noise:>5}")
print()

sft_usable = golden + non_golden_positive + non_golden_neutral
# Check whether non-golden docs have the message fields _hit_to_interaction needs
with_user_msg = count({"bool": {"filter": [
    {"term": {"data.type": "conversation"}},
    {"exists": {"field": "meta.rlhfl_evaluated_at"}},
    {"term": {"meta.rlhfl_is_golden": False}},
    {"term": {"meta.rlhfl_is_noise": False}},
    {"exists": {"field": "meta.rlhfl_user_message"}},
    {"range": {"@timestamp": {"gte": cutoff}}},
]}})
with_asst_resp = count({"bool": {"filter": [
    {"term": {"data.type": "conversation"}},
    {"exists": {"field": "meta.rlhfl_evaluated_at"}},
    {"term": {"meta.rlhfl_is_golden": False}},
    {"term": {"meta.rlhfl_is_noise": False}},
    {"exists": {"field": "meta.rlhfl_assistant_response"}},
    {"range": {"@timestamp": {"gte": cutoff}}},
]}})
with_both = count({"bool": {"filter": [
    {"term": {"data.type": "conversation"}},
    {"exists": {"field": "meta.rlhfl_evaluated_at"}},
    {"term": {"meta.rlhfl_is_golden": False}},
    {"term": {"meta.rlhfl_is_noise": False}},
    {"exists": {"field": "meta.rlhfl_user_message"}},
    {"exists": {"field": "meta.rlhfl_assistant_response"}},
    {"range": {"@timestamp": {"gte": cutoff}}},
]}})

print(f"  rlhfl_user_message present:        {with_user_msg:>5}  (of {non_golden_recent})")
print(f"  rlhfl_assistant_response present:  {with_asst_resp:>5}  (of {non_golden_recent})")
print(f"  both present (_hit_to_interaction): {with_both:>5}  → these actually make it to training")
print()

# Sample one doc that's missing the fields, show what's actually there
if with_both < non_golden_recent:
    sample = mgr.client.search(index=idx, query={"bool": {
        "filter": [
            {"term": {"data.type": "conversation"}},
            {"exists": {"field": "meta.rlhfl_evaluated_at"}},
            {"term": {"meta.rlhfl_is_golden": False}},
            {"term": {"meta.rlhfl_is_noise": False}},
            {"range": {"@timestamp": {"gte": cutoff}}},
        ],
        "must_not": [
            {"exists": {"field": "meta.rlhfl_user_message"}},
        ],
    }}, size=1)
    hits = sample["hits"]["hits"]
    if hits:
        src = hits[0]["_source"]
        print("Sample doc missing rlhfl_user_message — actual meta keys:")
        meta_keys = sorted(src.get("meta", {}).keys())
        print(f"  meta: {meta_keys}")
        print(f"  data keys: {sorted(src.get('data', {}).keys())}")

print(f"SFT-usable total: {sft_usable}  (golden {golden} + positive {non_golden_positive} + neutral {non_golden_neutral})")
print(f"Actually reachable by _hit_to_interaction: {with_both + golden}")
print()

if non_golden_recent == 0 and non_golden_all > 0:
    print("DIAGNOSIS: age cutoff — all non-golden interactions are older than the cutoff.")
    print(f"FIX: raise max_memory_age_days in system_config.yaml (currently {max_age}d).")
elif non_golden_negative == non_golden_recent and non_golden_recent > 0:
    print("DIAGNOSIS: ALL non-golden interactions are negative-sentiment — dropped by SFT.")
    print("  format_for_training() keeps only chosen≠None; negative examples set chosen=None")
    print("  (they're anti-patterns reserved for DPO, not used in SFT).")
    print()
    print("OPTIONS:")
    print("  1. Collect more positive/neutral interactions to train on.")
    print("  2. Switch to DPO mode once enough positive+negative pairs exist.")
    print("  3. Lower the negative sentiment threshold in ElasticsearchConfig")
    print("     so more interactions are classified as neutral rather than negative.")
elif sft_usable == golden:
    print(f"DIAGNOSIS: {non_golden_negative} negative interactions dropped by SFT → trained on golden only.")
    print("  See above — same fix applies (more positives, or DPO mode).")
else:
    print(f"DIAGNOSIS: {sft_usable} samples are SFT-usable. If training still only used {golden},")
    print("  check the 'Dataset composition' log line in rlhfl logs for that training run.")
