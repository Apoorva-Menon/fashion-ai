import pytest
from pathlib import Path
from fashion_ai.agents.skill_manager import SkillManager
from fashion_ai.pipeline import FashionAIPipeline
from fashion_ai.schemas.user_profile import PerspectiveImages, UserMetrics, UserProfile
from fashion_ai.schemas.tryon import GarmentArtifact

def test_skill_manager_resolution():
    sm = SkillManager()
    text, activated = sm.load_skills("hourglass", "deep_autumn")
    assert "hourglass.md" in activated
    assert "deep_autumn.md" in activated
    assert "=== ACTIVATED KNOWLEDGE: hourglass.md ===" in text

def test_end_to_end_mock_pipeline(tmp_path):
    pipeline = FashionAIPipeline(output_dir=tmp_path, mock_mode=True)

    profile = UserProfile(
        profile_id="test_user",
        images=PerspectiveImages(
            front_image_path="data/inputs/sample_front.jpg",
            side_image_path="data/inputs/sample_side.jpg",
        ),
        metrics=UserMetrics(
            height_cm=170.0,
            weight_kg=65.0,
            reported_skin_color="Warm golden",
            gender_expression="feminine",
        ),
    )

    garment = GarmentArtifact(
        artifact_id="hoodie_test",
        name="Oversized Hoodie with Baggy Pants",
        category="streetwear",
        color="Charcoal Gray",
        fabric="Fleece cotton",
        silhouette="Extreme oversized boxy cut",
    )

    results = pipeline.run(profile=profile, garment_artifact=garment)

    assert results["profile_id"] == "test_user"
    assert results["body_features"] is not None
    assert results["stylist_recommendations"] is not None
    assert results["tryon_verdict"] is not None

    # Check that output files were generated
    assert Path(results["body_features_file"]).exists()
    assert Path(results["recommendations_file"]).exists()
    assert results["preview_image"] is not None
    assert Path(results["preview_image"]).exists()
    assert results["subtitles_file"] is not None
    assert Path(results["subtitles_file"]).exists()
