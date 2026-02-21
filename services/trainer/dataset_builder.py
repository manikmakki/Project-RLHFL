import json
import logging
import random
from typing import List, Dict, Any
from datetime import datetime
from shared.models import Interaction, TrainingDatasetSample
from shared.config import SystemConfig

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Build training datasets from interactions with proper weighting."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
    
    def build_training_dataset(
        self,
        interactions: List[Interaction],
        golden_examples: List[Interaction]
    ) -> List[Dict[str, Any]]:
        """
        Build a training dataset from interactions and golden examples.
        
        Returns DPO-style dataset with chosen/rejected pairs.
        """
        logger.info(
            f"Building dataset from {len(interactions)} interactions "
            f"and {len(golden_examples)} golden examples"
        )
        
        # Separate by sentiment
        positive = [i for i in interactions if i.sentiment > self.config.sentiment.positive_threshold]
        negative = [i for i in interactions if i.sentiment < self.config.sentiment.negative_threshold]
        neutral = [
            i for i in interactions
            if abs(i.sentiment) <= max(
                self.config.sentiment.positive_threshold,
                abs(self.config.sentiment.negative_threshold)
            )
        ]
        
        logger.info(
            f"Sentiment distribution - Positive: {len(positive)}, "
            f"Negative: {len(negative)}, Neutral: {len(neutral)}"
        )
        
        dataset = []
        
        # Add positive examples (chosen responses)
        for interaction in positive:
            dataset.append({
                "prompt": interaction.user_message,
                "chosen": interaction.assistant_response,
                "rejected": None,  # Will be handled during training
                "weight": interaction.weight,
                "sentiment": interaction.sentiment,
                "type": "positive"
            })
        
        # Negative examples: stored as anti-patterns (rejected responses).
        # For SFT these are filtered out in format_for_training (chosen=None).
        # Retained in the dataset for future DPO support.
        for interaction in negative:
            dataset.append({
                "prompt": interaction.user_message,
                "chosen": None,
                "rejected": interaction.assistant_response,
                "weight": interaction.weight * 2.0,
                "sentiment": interaction.sentiment,
                "type": "negative"
            })
        
        # Include neutral examples to maintain breadth and allow emergent traits.
        # Interactions are already curated (top-N by weight), so include all neutral.
        for interaction in neutral:
            dataset.append({
                "prompt": interaction.user_message,
                "chosen": interaction.assistant_response,
                "rejected": None,
                "weight": interaction.weight * 0.5,  # Lower weight
                "sentiment": interaction.sentiment,
                "type": "neutral"
            })
        
        # Add golden examples (high importance)
        for example in golden_examples:
            dataset.append({
                "prompt": example.user_message,
                "chosen": example.assistant_response,
                "rejected": None,
                "weight": example.weight,
                "sentiment": example.sentiment,
                "type": "golden"
            })
        
        logger.info(f"Built dataset with {len(dataset)} total samples")

        # Log dataset composition
        type_counts = {}
        for sample in dataset:
            sample_type = sample['type']
            type_counts[sample_type] = type_counts.get(sample_type, 0) + 1

        sft_usable = sum(1 for s in dataset if s['chosen'] is not None)
        logger.info(f"Dataset composition: {type_counts} (SFT-usable: {sft_usable}, negative reserved for DPO: {type_counts.get('negative', 0)})")
        
        return dataset
    
    def create_validation_split(
        self,
        dataset: List[Dict[str, Any]],
        validation_split: float = 0.1
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split dataset into training and validation sets."""
        
        # Shuffle dataset
        shuffled = dataset.copy()
        random.shuffle(shuffled)
        
        # Split
        split_idx = int(len(shuffled) * (1 - validation_split))
        train_data = shuffled[:split_idx]
        val_data = shuffled[split_idx:]
        
        logger.info(
            f"Split dataset: {len(train_data)} training, {len(val_data)} validation"
        )
        
        return train_data, val_data
    
    def build_dpo_dataset(
        self,
        interactions: List[Interaction],
        golden_examples: List[Interaction],
        rejections: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Build DPO dataset with chosen/rejected pairs.

        Creates preference pairs from three sources:
        1. Positive/golden interactions paired with synthetic rejections
        2. Refusal interactions: the refusal text is "rejected", the retry
           response (if available) is "chosen"
        3. Neutral interactions paired with synthetic rejections

        Args:
            interactions: Top-N weighted interactions
            golden_examples: Golden examples (all included for catastrophic forgetting prevention)
            rejections: Map of prompt -> synthetic rejected response

        Returns:
            Dataset with (prompt, chosen, rejected) triplets for DPO training
        """
        # Separate refusal interactions (stored by psyche post-process)
        refusal_interactions = [i for i in interactions if i.metadata.get('is_refusal', False)]
        non_refusal_interactions = [i for i in interactions if not i.metadata.get('is_refusal', False)]

        logger.info(
            f"Building DPO dataset from {len(non_refusal_interactions)} interactions, "
            f"{len(refusal_interactions)} refusal interactions, "
            f"{len(golden_examples)} golden examples, "
            f"{len(rejections)} synthetic rejections"
        )

        dataset = []

        # Separate non-refusal interactions by sentiment
        positive = [i for i in non_refusal_interactions if i.sentiment > self.config.sentiment.positive_threshold]
        negative = [i for i in non_refusal_interactions if i.sentiment < self.config.sentiment.negative_threshold]
        neutral = [
            i for i in non_refusal_interactions
            if abs(i.sentiment) <= max(
                self.config.sentiment.positive_threshold,
                abs(self.config.sentiment.negative_threshold)
            )
        ]

        logger.info(
            f"Sentiment distribution - Positive: {len(positive)}, "
            f"Negative: {len(negative)}, Neutral: {len(neutral)}"
        )

        # Refusal interactions: the stored response IS the refusal (rejected).
        # If a retry succeeded, psyche_metadata contains the good response (chosen).
        for interaction in refusal_interactions:
            psyche_meta = interaction.psyche_metadata or {}
            retry_response = psyche_meta.get("retry_response")

            if retry_response:
                # Best case: we have both the refusal (rejected) and retry (chosen)
                dataset.append({
                    "prompt": interaction.user_message,
                    "chosen": retry_response,
                    "rejected": interaction.assistant_response,
                    "weight": interaction.weight,
                    "sentiment": interaction.sentiment,
                    "type": "refusal_dpo"
                })
            else:
                # No retry response available - store as rejected-only for now.
                # Can be paired with synthetic chosen responses in the future.
                dataset.append({
                    "prompt": interaction.user_message,
                    "chosen": None,
                    "rejected": interaction.assistant_response,
                    "weight": interaction.weight,
                    "sentiment": interaction.sentiment,
                    "type": "refusal_rejected_only"
                })

        logger.info(
            f"Refusal DPO pairs: {sum(1 for d in dataset if d['type'] == 'refusal_dpo')}, "
            f"Refusal rejected-only: {sum(1 for d in dataset if d['type'] == 'refusal_rejected_only')}"
        )

        # Positive examples with synthetic rejections (DPO pairs)
        for interaction in positive:
            rejection = rejections.get(interaction.user_message)
            if rejection:
                dataset.append({
                    "prompt": interaction.user_message,
                    "chosen": interaction.assistant_response,
                    "rejected": rejection,
                    "weight": interaction.weight,
                    "sentiment": interaction.sentiment,
                    "type": "positive_dpo"
                })
            else:
                # Fallback: if no rejection available, treat as SFT sample
                logger.warning(f"No rejection found for positive interaction, using SFT fallback")
                dataset.append({
                    "prompt": interaction.user_message,
                    "chosen": interaction.assistant_response,
                    "rejected": None,
                    "weight": interaction.weight,
                    "sentiment": interaction.sentiment,
                    "type": "positive_sft"
                })

        # Neutral examples with synthetic rejections
        for interaction in neutral:
            rejection = rejections.get(interaction.user_message)
            if rejection:
                dataset.append({
                    "prompt": interaction.user_message,
                    "chosen": interaction.assistant_response,
                    "rejected": rejection,
                    "weight": interaction.weight * 0.5,  # Lower weight
                    "sentiment": interaction.sentiment,
                    "type": "neutral_dpo"
                })

        # Golden examples with synthetic rejections (CRITICAL - prevents catastrophic forgetting)
        for example in golden_examples:
            rejection = rejections.get(example.user_message)
            if rejection:
                dataset.append({
                    "prompt": example.user_message,
                    "chosen": example.assistant_response,
                    "rejected": rejection,
                    "weight": example.weight * 2.0,  # Higher weight for golden examples
                    "sentiment": example.sentiment,
                    "type": "golden_dpo"
                })
            else:
                logger.warning(f"No rejection found for golden example, using SFT fallback")
                dataset.append({
                    "prompt": example.user_message,
                    "chosen": example.assistant_response,
                    "rejected": None,
                    "weight": example.weight * 2.0,
                    "sentiment": example.sentiment,
                    "type": "golden_sft"
                })

        # Negative examples (non-refusal): use actual bad responses as rejections
        # Pair with similar positive response or skip if no good alternative
        for interaction in negative:
            # For now, skip negative-only examples in DPO
            # Future: could pair with similar positive responses
            logger.debug(f"Skipping negative-only interaction (no chosen alternative)")

        logger.info(f"Built DPO dataset with {len(dataset)} total samples")

        # Log dataset composition
        type_counts = {}
        for sample in dataset:
            sample_type = sample['type']
            type_counts[sample_type] = type_counts.get(sample_type, 0) + 1

        dpo_pairs = sum(1 for s in dataset if s.get('rejected') is not None)
        logger.info(
            f"DPO dataset composition: {type_counts} "
            f"(DPO pairs: {dpo_pairs}, SFT fallbacks: {len(dataset) - dpo_pairs})"
        )

        return dataset

    def format_for_training(
        self,
        dataset: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Format dataset for SFT training framework.

        Filters to only chosen responses, discarding rejected responses.
        """
        formatted = []

        for sample in dataset:
            # For supervised fine-tuning, use only chosen responses
            if sample['chosen'] is not None:
                formatted.append({
                    "prompt": sample['prompt'],
                    "completion": sample['chosen'],
                    "weight": sample['weight']
                })

        return formatted

    def format_for_dpo_training(
        self,
        dataset: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Format dataset for DPO training framework.

        Returns only samples with both chosen and rejected responses.
        Samples without rejected responses are skipped (DPO requires pairs).
        """
        formatted = []

        for sample in dataset:
            # DPO requires both chosen and rejected responses
            if sample.get('chosen') and sample.get('rejected'):
                formatted.append({
                    "prompt": sample['prompt'],
                    "chosen": sample['chosen'],
                    "rejected": sample['rejected']
                })

        logger.info(
            f"Formatted {len(formatted)} DPO pairs "
            f"({len(dataset) - len(formatted)} samples skipped - missing pairs)"
        )

        return formatted
