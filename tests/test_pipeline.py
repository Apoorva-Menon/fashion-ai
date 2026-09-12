import pytest
from pathlib import Path
from PIL import Image
import numpy as np

from fashion_ai.agents.skill_manager import SkillManager
from fashion_ai.services.vision_service import VisionService
from fashion_ai.pipeline import FashionAIPipeline
from fashion_ai.schemas.user_profile import PerspectiveImages, UserMetrics, UserProfile
from fashion_ai.schemas.tryon import GarmentArtifact

def test_skill_manager_resolution():
    sm = SkillManager()
    text, activated = sm.load_skills("hourglass", "deep_autumn")
    assert "hourglass.md" in activated
    assert "deep_autumn.md" in activated
    assert "=== ACTIVATED KNOWLEDGE: hourglass.md ===" in text

def test_vision_service_live_frame_analysis():
    vs = VisionService(mock_mode=True)
    # Simulate a 400x400x3 numpy frame as captured from live video
    mock_frame = np.zeros((400, 400, 3), dtype=np.uint8)
    
    result = vs.analyze_frame(mock_frame, perspective="front")
    assert result["perspective"] == "front"
    assert "dominant_colors" in result
    assert "face_annotations" in result
    assert "detected_objects" in result
    assert len(result["dominant_colors"]) > 0

def test_vision_service_multi_perspectives():
    vs = VisionService(mock_mode=True)
    perspectives = {
        "front": "data/inputs/sample_front.jpg",
        "side": "data/inputs/sample_side.jpg",
    }
    report = vs.analyze_perspectives(perspectives)
    assert report["total_perspectives_analyzed"] >= 1
    assert "composite_dominant_colors" in report
    assert "detected_objects_summary" in report

def test_end_to_end_mock_pipeline_with_live_vision(tmp_path):
    pipeline = FashionAIPipeline(output_dir=tmp_path, mock_mode=True)

    profile = UserProfile(
        profile_id="test_user_live",
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

    # Simulate signals captured during live video session
    live_signals = {
        "front": {
            "dominant_colors": [{"hex": "#d4a373", "score": 0.5}],
            "face_annotations": [{"roll_angle": 0.5, "detection_confidence": 0.99}],
            "detected_objects": [{"name": "Person", "score": 0.98}],
        }
    }

    results = pipeline.run(
        profile=profile,
        garment_artifact=garment,
        precomputed_vision_signals=live_signals,
    )

    assert results["profile_id"] == "test_user_live"
    assert results["body_features"] is not None
    assert results["stylist_recommendations"] is not None
    assert results["tryon_verdict"] is not None
    assert Path(results["vision_analysis_file"]).exists()
    assert Path(results["body_features_file"]).exists()
    assert Path(results["recommendations_file"]).exists()
    assert Path(results["preview_image"]).exists()
    assert Path(results["subtitles_file"]).exists()

def test_vonage_video_service(tmp_path):
    from fashion_ai.services.vonage_video_service import VonageVideoService
    v_service = VonageVideoService(mock_mode=True)

    session = v_service.create_video_session()
    assert "session_id" in session
    assert session["is_mock"] is True

    token = v_service.generate_client_token(session["session_id"])
    assert token.startswith("T1==")

    html_file = v_service.generate_webrtc_html_client(
        session_id=session["session_id"],
        token=token,
        output_file=tmp_path / "test_client.html",
    )
    assert html_file.exists()
    assert "Fashion AI - Vonage Live Video Room" in html_file.read_text()
