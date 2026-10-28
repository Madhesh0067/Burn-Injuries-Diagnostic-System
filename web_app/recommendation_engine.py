"""
Ayurvedic Recommendation Engine for Burn Injuries Diagnostic System.
Maps AI diagnostic burn classes to classical Ayurvedic treatment protocols,
dosha balancing principles, topical remedies, and modern clinical emergency triage.
"""

import json
import os
from typing import Dict, Any, Optional

KNOWLEDGE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "ayurvedic_knowledge.json"
)

class AyurvedicRecommendationEngine:
    def __init__(self, knowledge_path: Optional[str] = None):
        self.knowledge_path = knowledge_path or KNOWLEDGE_PATH
        self.knowledge_base = self._load_knowledge()

    def _load_knowledge(self) -> Dict[str, Any]:
        if os.path.exists(self.knowledge_path):
            with open(self.knowledge_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def get_recommendations(self, predicted_label: str, confidence: float) -> Dict[str, Any]:
        """
        Normalizes label and returns structured Ayurvedic & clinical recommendations.
        """
        # Normalize key
        norm_key = predicted_label.lower().replace(" ", "_").replace("-", "_")
        if "1" in norm_key or "first" in norm_key:
            key = "degree_1"
        elif "2" in norm_key or "second" in norm_key:
            key = "degree_2"
        elif "3" in norm_key or "third" in norm_key:
            key = "degree_3"
        elif "health" in norm_key or "normal" in norm_key:
            key = "healthy"
        else:
            key = "degree_1"

        data = self.knowledge_base.get(key, {})
        if not data:
            return {
                "title": predicted_label,
                "severity_level": "Unknown",
                "dosha_involvement": "Consult an Ayurvedic physician",
                "clinical_presentation": "Assessment pending.",
                "immediate_first_aid": [],
                "ayurvedic_topical_therapies": [],
                "dietary_guidelines": [],
                "safety_precautions": ["Seek direct medical evaluation."]
            }

        result = dict(data)
        result["confidence"] = round(confidence * 100, 2)
        result["class_key"] = key
        result["is_emergency"] = (key == "degree_3")
        return result
