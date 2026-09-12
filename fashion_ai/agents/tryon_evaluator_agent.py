import json
import logging
from typing import Optional
from fashion_ai.config import GEMINI_API_KEY, TRYON_MODEL, MOCK_MODE
from fashion_ai.schemas.body_features import BodyFeatures
from fashion_ai.schemas.stylist_recommendations import StylistRecommendations
from fashion_ai.schemas.tryon import GarmentArtifact, TryOnVerdict
from fashion_ai.prompts.tryon import TRYON_SYSTEM_PROMPT, TRYON_USER_PROMPT

logger = logging.getLogger(__name__)

class TryOnEvaluatorAgent:
    """Evaluates clothing artifact against user body profile and outputs try-on verdict & subtitles."""

    def __init__(
        self,
        model_name: str = TRYON_MODEL,
        api_key: Optional[str] = None,
        mock_mode: bool = MOCK_MODE,
    ):
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

    def evaluate_garment(
        self,
        features: BodyFeatures,
        recommendations: StylistRecommendations,
        artifact: GarmentArtifact,
    ) -> TryOnVerdict:
        """Evaluates garment suitability and generates try-on commentary."""
        if self.mock_mode or not self._client:
            return self._mock_evaluate(features, recommendations, artifact)

        from google import genai
        from google.genai import types

        prompt = TRYON_USER_PROMPT.format(
            body_features_json=features.model_dump_json(indent=2),
            stylist_rules_json=recommendations.model_dump_json(indent=2),
            artifact_id=artifact.artifact_id,
            garment_name=artifact.name,
            category=artifact.category,
            color=artifact.color,
            fabric=artifact.fabric,
            silhouette=artifact.silhouette,
            image_path=artifact.image_path or "None",
        )

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=[prompt],
            config=types.GenerateContentConfig(
                system_instruction=TRYON_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=TryOnVerdict,
                temperature=0.2,
            ),
        )

        return TryOnVerdict.model_validate_json(response.text)

    def _mock_evaluate(
        self,
        features: BodyFeatures,
        recommendations: StylistRecommendations,
        artifact: GarmentArtifact,
    ) -> TryOnVerdict:
        """Provides realistic mock evaluation for garments like 'Hoodie with Baggy Pants'."""
        name_lower = artifact.name.lower()
        shape = features.body_shape

        # If it's oversized hoodie with baggy pants on an inverted triangle or hourglass:
        is_baggy_hoodie = "hoodie" in name_lower or "baggy" in name_lower or "oversized" in artifact.silhouette.lower()

        if is_baggy_hoodie and shape in ("inverted_triangle", "hourglass", "apple"):
            return TryOnVerdict(
                artifact_id=artifact.artifact_id,
                garment_name=artifact.name,
                verdict="not_good",
                score_out_of_10=3.5,
                verdict_headline="Avoid: Heavy drop-shoulder hoodie and baggy pants obscure waist and exaggerate upper frame width.",
                detailed_rationale=(
                    f"For an individual with a {shape} silhouette, pairing an oversized heavyweight fleece hoodie "
                    f"with shapeless baggy pants creates double-volume. It obliterates your natural waistline, "
                    f"shortens your perceived leg length, and makes the torso appear broader than it actually is."
                ),
                how_to_style_or_fix=[
                    "Swap the heavy oversized hoodie for a fitted zip cardigan or open lightweight overshirt.",
                    "If keeping the baggy pants, pair with a clean, form-fitting ribbed tank or tucked-in tee.",
                    "Roll or hem pants at the ankle to reveal footwear and break the continuous bulk.",
                ],
                tryon_video_subtitle_script=(
                    f"Looking at the live camera feed with the {artifact.name}: Notice how the heavy fabric "
                    f"and drop-shoulder cut drowns your natural structure. Because your silhouette is {shape}, "
                    f"this double-baggy combination makes your proportions look bottom-heavy and unanchored. "
                    f"Rating: 3.5 out of 10. Verdict: Not recommended in this configuration."
                ),
            )
        else:
            return TryOnVerdict(
                artifact_id=artifact.artifact_id,
                garment_name=artifact.name,
                verdict="good",
                score_out_of_10=8.5,
                verdict_headline=f"Excellent Match: Complements your {shape} frame and color palette.",
                detailed_rationale=(
                    f"This garment cut harmonizes with your {shape} body lines, accentuating your vertical proportions "
                    f"while maintaining comfortable, elegant ease."
                ),
                how_to_style_or_fix=[
                    "Pair with minimalist leather sneakers or clean dress boots.",
                    "Accompany with subtle warm gold accessories."
                ],
                tryon_video_subtitle_script=(
                    f"Live virtual try-on verdict for {artifact.name}: This silhouette flows cleanly along your vertical lines. "
                    f"The fabric drapes naturally without bunching, creating a balanced, stylish impression. Score: 8.5/10."
                ),
            )
