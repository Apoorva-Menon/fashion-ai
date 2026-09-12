import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from fashion_ai.config import GEMINI_API_KEY, FEATURE_EXTRACTOR_MODEL, MOCK_MODE
from fashion_ai.schemas.user_profile import UserProfile
from fashion_ai.schemas.body_features import BodyFeatures, Proportions, ToneAnalysis, FacialStructure
from fashion_ai.prompts.feature_extraction import (
    FEATURE_EXTRACTION_SYSTEM_PROMPT,
    FEATURE_EXTRACTION_USER_PROMPT,
)

logger = logging.getLogger(__name__)

class FeatureExtractorAgent:
    """Multimodal agent analyzing user multi-perspective images and metrics."""

    def __init__(self, model_name: str = FEATURE_EXTRACTOR_MODEL, api_key: Optional[str] = None, mock_mode: bool = MOCK_MODE):
        self.model_name = model_name
        self.api_key = api_key or GEMINI_API_KEY
        self.mock_mode = mock_mode
        self._client = None

        if not self.mock_mode and self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize google-genai client: {e}. Defaulting to mock mode.")
                self.mock_mode = True
        else:
            self.mock_mode = True

    def extract_features(self, profile: UserProfile, vision_signals: Optional[Dict[str, Any]] = None) -> BodyFeatures:
        """Extracts structured BodyFeatures using multimodal Gemini + Vision cues."""
        if self.mock_mode or not self._client:
            return self._mock_extract(profile, vision_signals)

        from google import genai
        from google.genai import types
        from PIL import Image

        prompt_text = FEATURE_EXTRACTION_USER_PROMPT.format(
            height_cm=profile.metrics.height_cm,
            weight_kg=profile.metrics.weight_kg,
            reported_skin_color=profile.metrics.reported_skin_color,
            gender_expression=profile.metrics.gender_expression or "unspecified",
            vision_signals=json.dumps(vision_signals or {}, indent=2),
        )

        contents = [prompt_text]

        # Attach multi-perspective images if present
        for img_path in [profile.images.front_image_path, profile.images.side_image_path, profile.images.angled_image_path]:
            if img_path and Path(img_path).exists():
                try:
                    pil_img = Image.open(img_path)
                    contents.append(pil_img)
                except Exception as e:
                    logger.warning(f"Could not load image {img_path}: {e}")

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=FEATURE_EXTRACTION_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=BodyFeatures,
                temperature=0.2,
            )
        )

        return BodyFeatures.model_validate_json(response.text)

    def _mock_extract(self, profile: UserProfile, vision_signals: Optional[Dict[str, Any]]) -> BodyFeatures:
        """Realistic mock extraction based on user metrics for testing."""
        # Simple heuristic for mock diversity
        bmi = profile.metrics.weight_kg / ((profile.metrics.height_cm / 100) ** 2)
        shape = "hourglass" if profile.metrics.gender_expression == "feminine" else "inverted_triangle"
        
        return BodyFeatures(
            body_shape=shape,
            color_season="deep_autumn",
            proportions=Proportions(
                shoulder_to_hip_ratio="Athletic upper body with well-aligned shoulders",
                torso_to_leg_ratio="Balanced 1:1 torso-to-leg proportion",
                vertical_line="Moderate to elongated vertical perception",
                waist_definition="Moderately tapered waist indent",
            ),
            tone=ToneAnalysis(
                undertone="warm",
                contrast_level="medium",
                dominant_hex_colors=["#d4a373", "#3d2b1f", "#2b2d42"],
                recommended_season="deep_autumn",
            ),
            face=FacialStructure(
                face_shape="oval",
                jawline="softly defined",
            ),
            overall_summary=(
                f"Calibrated for height {profile.metrics.height_cm}cm, weight {profile.metrics.weight_kg}kg "
                f"(BMI ~{bmi:.1f}). Characterized by a {shape} silhouette with warm deep autumn undertones."
            )
        )
