import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from fashion_ai.config import OUTPUT_DIR, MOCK_MODE
from fashion_ai.schemas.user_profile import UserProfile
from fashion_ai.schemas.body_features import BodyFeatures
from fashion_ai.schemas.stylist_recommendations import StylistRecommendations
from fashion_ai.schemas.tryon import GarmentArtifact, TryOnVerdict
from fashion_ai.services.vision_service import VisionService
from fashion_ai.services.tryon_service import TryOnService
from fashion_ai.agents.skill_manager import SkillManager
from fashion_ai.agents.feature_extractor_agent import FeatureExtractorAgent
from fashion_ai.agents.stylist_agent import StylistAgent
from fashion_ai.agents.tryon_evaluator_agent import TryOnEvaluatorAgent

logger = logging.getLogger(__name__)

class FashionAIPipeline:
    """End-to-end pipeline coordinating Vision, Multimodal Feature Extraction, Skill-based Styling, and Try-On."""

    def __init__(
        self,
        output_dir: Path = OUTPUT_DIR,
        mock_mode: bool = MOCK_MODE,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mock_mode = mock_mode

        # Initialize Services
        self.vision_service = VisionService(mock_mode=self.mock_mode)
        self.tryon_service = TryOnService(output_dir=self.output_dir)
        self.skill_manager = SkillManager()

        # Initialize Agents
        self.feature_agent = FeatureExtractorAgent(mock_mode=self.mock_mode)
        self.stylist_agent = StylistAgent(mock_mode=self.mock_mode, skill_manager=self.skill_manager)
        self.tryon_agent = TryOnEvaluatorAgent(mock_mode=self.mock_mode)

    def run(
        self,
        profile: UserProfile,
        garment_artifact: Optional[GarmentArtifact] = None,
        precomputed_vision_signals: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Executes the full styling and try-on pipeline with live/file Vision API analysis."""
        logger.info(f"Starting Fashion AI pipeline for profile: {profile.profile_id}")

        # -------------------------------------------------------------
        # 1. Multi-Perspective Vision Analysis (Google Cloud Vision API)
        # -------------------------------------------------------------
        if precomputed_vision_signals:
            logger.info("Stage 1: Using Vision API signals captured during live video session.")
            vision_signals = {
                "perspectives": precomputed_vision_signals,
                "total_perspectives_analyzed": len(precomputed_vision_signals),
                "is_live_capture": True,
            }
        else:
            logger.info("Stage 1: Extracting Vision API cues across all available perspective images...")
            vision_signals = self.vision_service.analyze_perspectives({
                "front": profile.images.front_image_path,
                "side": profile.images.side_image_path,
                "angled": profile.images.angled_image_path,
            })

        # Persist Vision API findings
        vision_file = self.output_dir / f"{profile.profile_id}_vision_analysis.json"
        vision_file.write_text(json.dumps(vision_signals, indent=2), encoding="utf-8")
        logger.info(f"Saved Vision API analysis to {vision_file}")

        # -------------------------------------------------------------
        # 2. Body Feature Extraction (Gemini Multimodal + Metrics + Vision)
        # -------------------------------------------------------------
        logger.info("Stage 2: Extracting structural body features & color season via LLM Agent...")
        features: BodyFeatures = self.feature_agent.extract_features(profile, vision_signals=vision_signals)
        features_file = self.output_dir / f"{profile.profile_id}_body_features.json"
        features_file.write_text(features.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"Saved body features to {features_file}")

        # -------------------------------------------------------------
        # 3. Stylist Recommender (Skill-Powered Agent)
        # -------------------------------------------------------------
        logger.info(f"Stage 3: Loading skills for {features.body_shape} & {features.color_season}...")
        recommendations: StylistRecommendations = self.stylist_agent.generate_recommendations(features)
        recs_file = self.output_dir / f"{profile.profile_id}_stylist_recommendations.json"
        recs_file.write_text(recommendations.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"Saved stylist recommendations to {recs_file}")

        # -------------------------------------------------------------
        # 4. Garment Artifact Evaluation & Try-On (if provided)
        # -------------------------------------------------------------
        tryon_verdict = None
        preview_image_path = None
        subtitles_file_path = None

        if garment_artifact:
            logger.info(f"Stage 4: Evaluating garment artifact: {garment_artifact.name}...")
            tryon_verdict = self.tryon_agent.evaluate_garment(features, recommendations, garment_artifact)

            # Generate visual comparison preview
            preview_image_path = self.tryon_service.generate_tryon_preview(
                user_image_path=profile.images.front_image_path,
                artifact=garment_artifact,
                verdict=tryon_verdict,
            )
            tryon_verdict.preview_output_path = preview_image_path

            # Export video subtitle file (.srt)
            subtitles_file_path = self.tryon_service.export_subtitles(tryon_verdict)

            verdict_file = self.output_dir / f"{profile.profile_id}_tryon_{garment_artifact.artifact_id}.json"
            verdict_file.write_text(tryon_verdict.model_dump_json(indent=2), encoding="utf-8")
            logger.info(f"Saved tryon verdict to {verdict_file}")

        return {
            "profile_id": profile.profile_id,
            "vision_signals": vision_signals,
            "vision_analysis_file": str(vision_file),
            "body_features": features,
            "body_features_file": str(features_file),
            "stylist_recommendations": recommendations,
            "recommendations_file": str(recs_file),
            "tryon_verdict": tryon_verdict,
            "preview_image": preview_image_path,
            "subtitles_file": subtitles_file_path,
        }
