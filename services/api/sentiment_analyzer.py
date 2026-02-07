import re
import logging
from typing import Optional
from shared.models import ChatMessage
from shared.config import SystemConfig

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Infer sentiment from user messages using hybrid approach."""
    
    # Strong indicators
    STRONG_POSITIVE = [
        r'\b(thank(?:s| you)|perfect|excellent|amazing|brilliant|fantastic)\b',
        r'\b(exactly what i (?:wanted|needed))\b',
        r'\b(love it|nailed it|spot on)\b',
    ]
    
    STRONG_NEGATIVE = [
        r'\b(no+|wrong|incorrect|bad|awful|terrible)\b',
        r'\b(stop|don\'t do that|never do)\b',
        r'\b(try again|redo|not (?:right|correct|what i))\b',
    ]
    
    # Moderate indicators
    MODERATE_POSITIVE = [
        r'\b(good|great|nice|helpful|useful|appreciate)\b',
        r'\b(yes|correct|right|ok|okay)\b',
    ]
    
    MODERATE_NEGATIVE = [
        r'\b(not (?:good|great|helpful|right))\b',
        r'\b(issue|problem|error|mistake|fix|change)\b',
    ]
    
    # High-value instruction patterns
    INSTRUCTION_PATTERNS = [
        r'\b(always|never|remember|don\'t forget)\b',
        r'\b(important|crucial|critical|essential)\b',
        r'\b(from now on|going forward|in the future)\b',
        r'\b(prefer|want|need|should|must)\b',
    ]
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.positive_threshold = config.sentiment.positive_threshold
        self.negative_threshold = config.sentiment.negative_threshold
        self.instruction_boost = config.sentiment.explicit_instruction_boost
        self.continuation_baseline = config.sentiment.continuation_baseline
    
    def analyze(
        self,
        user_message: str,
        assistant_response: str,
        conversation_continuing: bool = True
    ) -> float:
        """
        Analyze sentiment using multiple signals.
        
        Returns:
            float: Sentiment score from -1.0 (very negative) to +1.0 (very positive)
        """
        text = user_message.lower()
        sentiment = 0.0
        signals = []
        
        # Signal 1: Strong explicit feedback (highest priority)
        strong_neg = sum(1 for p in self.STRONG_NEGATIVE if re.search(p, text))
        strong_pos = sum(1 for p in self.STRONG_POSITIVE if re.search(p, text))
        
        if strong_neg > 0:
            sentiment = -0.8 * min(strong_neg / 2.0, 1.0)
            signals.append(f"strong_negative({strong_neg})")
            logger.debug(f"Strong negative sentiment: {sentiment:.2f}")
            return sentiment
        
        if strong_pos > 0:
            sentiment = 0.8 * min(strong_pos / 2.0, 1.0)
            signals.append(f"strong_positive({strong_pos})")
            logger.debug(f"Strong positive sentiment: {sentiment:.2f}")
            return sentiment
        
        # Signal 2: Moderate feedback
        mod_neg = sum(1 for p in self.MODERATE_NEGATIVE if re.search(p, text))
        mod_pos = sum(1 for p in self.MODERATE_POSITIVE if re.search(p, text))
        
        if mod_neg > mod_pos:
            sentiment -= 0.4 * (mod_neg / (mod_neg + mod_pos + 1))
            signals.append(f"moderate_negative({mod_neg})")
        elif mod_pos > mod_neg:
            sentiment += 0.4 * (mod_pos / (mod_neg + mod_pos + 1))
            signals.append(f"moderate_positive({mod_pos})")
        
        # Signal 3: Message length (very short follow-ups often indicate issues)
        if len(user_message) < 10 and conversation_continuing:
            sentiment -= 0.1
            signals.append("very_short")
        
        # Signal 4: Question marks (multiple questions may indicate confusion)
        question_count = text.count('?')
        if question_count > 2:
            sentiment -= 0.15
            signals.append(f"many_questions({question_count})")
        
        # Signal 5: Exclamation marks (enthusiasm or frustration)
        exclamation_count = text.count('!')
        if exclamation_count > 0:
            # Context matters - combine with other signals
            if sentiment >= 0:
                sentiment += 0.1 * min(exclamation_count, 2)
                signals.append("enthusiastic")
            else:
                sentiment -= 0.1 * min(exclamation_count, 2)
                signals.append("frustrated")
        
        # Signal 6: Instructions (high value, neutral-to-positive)
        instruction_matches = sum(1 for p in self.INSTRUCTION_PATTERNS if re.search(p, text))
        if instruction_matches > 0:
            sentiment = max(sentiment, 0.3)  # Instructions are at least mildly positive
            signals.append(f"instruction({instruction_matches})")
        
        # Signal 7: Conversation continuation (baseline positive)
        if conversation_continuing and sentiment == 0:
            sentiment = self.continuation_baseline
            signals.append("continuation")
        
        # Clamp to [-1, 1]
        sentiment = max(-1.0, min(1.0, sentiment))
        
        logger.debug(f"Sentiment: {sentiment:.2f} | Signals: {', '.join(signals)}")
        return sentiment
    
    def calculate_weight(
        self,
        user_message: str,
        sentiment: float,
        conversation_continuing: bool = True
    ) -> float:
        """
        Calculate the weight/importance of this interaction.
        
        Returns:
            float: Weight multiplier (1.0 = normal, higher = more important)
        """
        weight = 1.0
        text = user_message.lower()
        
        # Boost 1: Explicit instructions
        instruction_matches = sum(1 for p in self.INSTRUCTION_PATTERNS if re.search(p, text))
        if instruction_matches > 0:
            weight *= self.instruction_boost
        
        # Boost 2: Strong sentiment (both positive and negative are important)
        weight *= (1.0 + abs(sentiment) * 0.5)
        
        # Boost 3: Message length (substantive interactions matter more)
        msg_length = len(user_message)
        if msg_length > 200:
            weight *= 1.5
        elif msg_length > 500:
            weight *= 2.0
        elif msg_length < 20:
            weight *= 0.7  # Very short messages less important
        
        # Boost 4: Specific directives
        if any(word in text for word in ['prefer', 'want', 'need', 'should', 'must', 'always', 'never']):
            weight *= 1.3
        
        # Cap maximum weight
        return min(weight, 5.0)
    
    def is_golden_example(
        self,
        user_message: str,
        sentiment: float,
        weight: float
    ) -> bool:
        """
        Determine if this interaction should be a golden example.
        """
        text = user_message.lower()
        
        # Criterion 1: High-weight interactions (important instructions)
        if weight >= 3.0:
            return True
        
        # Criterion 2: Strong explicit feedback
        if abs(sentiment) >= 0.7:
            return True
        
        # Criterion 3: Explicit preference/instruction statements
        preference_phrases = [
            'i prefer', 'i want', 'i need', 'i like when',
            'always do', 'never do', 'remember to', 'make sure',
            'from now on', 'going forward'
        ]
        if any(phrase in text for phrase in preference_phrases):
            return True
        
        # Criterion 4: Detailed corrections (long negative feedback)
        if sentiment < -0.3 and len(user_message) > 100:
            return True
        
        return False