import pytest
from fashion_ai.schemas.user_profile import PerspectiveImages, UserMetrics, UserProfile
from fashion_ai.schemas.body_features import BodyFeatures, Proportions, ToneAnalysis, FacialStructure
from fashion_ai.schemas.stylist_recommendations import (
    StylistRecommendations,
    BodyDosAndDonts,
    SkinAnalysisDosAndDonts,
    CategoryGuidance,
    SubtitleLine,
)
from fashion_ai.schemas.tryon import GarmentArtifact, TryOnVerdict

def test_user_profile_schema():
    profile = UserProfile(
        profile_id="test_001",
        images=PerspectiveImages(
            front_image_path="data/inputs/sample_front.jpg",
            side_image_path="data/inputs/sample_side.jpg",
        ),
        metrics=UserMetrics(
            height_cm=178.0,
            weight_kg=72.5,
            reported_skin_color="Fair with neutral undertones",
            gender_expression="masculine",
        ),
    )
    assert profile.metrics.height_cm == 178.0
    assert profile.images.side_image_path == "data/inputs/sample_side.jpg"

def test_body_features_schema():
    bf = BodyFeatures(
        body_shape="hourglass",
        color_season="warm_spring",
        proportions=Proportions(
            shoulder_to_hip_ratio="balanced",
            torso_to_leg_ratio="1:1 balanced",
            vertical_line="elongated",
            waist_definition="clearly defined",
        ),
        tone=ToneAnalysis(
            undertone="warm",
            contrast_level="medium",
            dominant_hex_colors=["#e0ac69", "#4a3525"],
            recommended_season="warm_spring",
        ),
        face=FacialStructure(face_shape="oval", jawline="soft"),
        overall_summary="Hourglass silhouette with warm spring coloration.",
    )
    data_json = bf.model_dump_json()
    assert "hourglass" in data_json
    reloaded = BodyFeatures.model_validate_json(data_json)
    assert reloaded.body_shape == "hourglass"

def test_stylist_recommendations_with_dos_and_donts():
    recs = StylistRecommendations(
        body_type_identified="inverted_triangle",
        color_season_identified="deep_autumn",
        activated_skills=["inverted_triangle.md", "deep_autumn.md"],
        body_dos_and_donts=BodyDosAndDonts(
            dos=["Wear wide leg pants to balance shoulders"],
            donts=["Avoid shoulder pads"],
            proportional_rationale="Balances upper athletic frame",
        ),
        skin_dos_and_donts=SkinAnalysisDosAndDonts(
            dos=["Wear warm olive and rich camel"],
            donts=["Avoid optical white"],
            chromatic_rationale="Warm undertone thrives on earthy pigments",
        ),
        what_suits_best=CategoryGuidance(
            fabrics=["Wool"],
            silhouettes_and_cuts=["Wide leg"],
            colors_and_palettes=["Olive"],
        ),
        what_to_avoid=CategoryGuidance(
            fabrics=["Stiff fleece"],
            silhouettes_and_cuts=["Baggy hoodie"],
            colors_and_palettes=["Ice pink"],
        ),
        key_rules=["Rule 1"],
        video_subtitles=[SubtitleLine(sequence=1, timestamp="00:00 - 00:05", narration="Welcome")],
    )
    assert len(recs.body_dos_and_donts.dos) == 1
    assert len(recs.skin_dos_and_donts.donts) == 1

def test_tryon_verdict_schema():
    verdict = TryOnVerdict(
        artifact_id="hoodie_baggy_pants",
        garment_name="Oversized Hoodie with Baggy Pants",
        verdict="not_good",
        score_out_of_10=4.0,
        verdict_headline="Avoid: Excess volume hides waist definition.",
        detailed_rationale="Overly boxy cut disrupts balanced proportions.",
        how_to_style_or_fix=["Tuck in", "Swap for tapered bottom"],
        tryon_video_subtitle_script="Looking at the live try-on, this outfit overwhelms your frame.",
    )
    assert verdict.verdict == "not_good"
    assert len(verdict.how_to_style_or_fix) == 2
