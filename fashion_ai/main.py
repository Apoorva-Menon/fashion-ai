import argparse
import json
import logging
import sys
from pathlib import Path

from fashion_ai.config import INPUTS_DIR, ARTIFACTS_DIR, OUTPUT_DIR, MOCK_MODE
from fashion_ai.schemas.user_profile import PerspectiveImages, UserMetrics, UserProfile
from fashion_ai.schemas.tryon import GarmentArtifact
from fashion_ai.services.camera import CameraCaptureService
from fashion_ai.pipeline import FashionAIPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fashion_ai.main")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fashion AI: Multi-Perspective Visual Styling & Virtual Try-On Pipeline"
    )
    parser.add_argument("--mock", action="store_true", default=MOCK_MODE, help="Run in offline mock mode")
    parser.add_argument("--camera", action="store_true", help="Launch interactive camera capture for front/side images")

    # Image inputs
    parser.add_argument("--front", type=str, help="Path to front full-body image")
    parser.add_argument("--side", type=str, help="Path to side profile full-body image")
    parser.add_argument("--angled", type=str, help="Path to angled 45-degree full-body image")

    # Ground truth metrics
    parser.add_argument("--height", type=float, default=175.0, help="User height in cm (default: 175)")
    parser.add_argument("--weight", type=float, default=70.0, help="User weight in kg (default: 70)")
    parser.add_argument("--skin-tone", type=str, default="Medium with warm olive undertones", help="Self-reported skin tone")
    parser.add_argument("--gender", type=str, default="unspecified", help="Styling gender expression (feminine, masculine, unisex)")

    # Garment Artifact
    parser.add_argument(
        "--artifact",
        type=str,
        default="data/artifacts/hoodie_with_baggy_pants.json",
        help="Path to garment artifact JSON to evaluate in try-on stage",
    )
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    front_img = args.front
    side_img = args.side
    angled_img = args.angled

    # If camera capture requested
    if args.camera:
        logger.info("Opening live webcam capture...")
        cam_service = CameraCaptureService(output_dir=INPUTS_DIR)
        captured = cam_service.capture_perspectives_interactive()
        front_img = captured.get("front", front_img)
        side_img = captured.get("side", side_img)
        angled_img = captured.get("angled", angled_img)

    # Fallback to sample placeholder if none provided
    if not front_img:
        sample_front = INPUTS_DIR / "sample_front.jpg"
        if sample_front.exists():
            front_img = str(sample_front)
        else:
            front_img = "mock_front.jpg"

    # Assemble UserProfile
    profile = UserProfile(
        profile_id="user_demo",
        images=PerspectiveImages(
            front_image_path=front_img,
            side_image_path=side_img,
            angled_image_path=angled_img,
        ),
        metrics=UserMetrics(
            height_cm=args.height,
            weight_kg=args.weight,
            reported_skin_color=args.skin_tone,
            gender_expression=args.gender,
        ),
    )

    # Load Garment Artifact
    garment = None
    artifact_path = Path(args.artifact)
    if not artifact_path.is_absolute():
        artifact_path = Path.cwd() / artifact_path

    if artifact_path.exists():
        try:
            data = json.loads(artifact_path.read_text(encoding="utf-8"))
            garment = GarmentArtifact(**data)
            logger.info(f"Loaded garment artifact: '{garment.name}' ({garment.artifact_id})")
        except Exception as e:
            logger.error(f"Failed to parse garment artifact JSON {artifact_path}: {e}")
    else:
        logger.warning(f"Garment artifact file not found at {artifact_path}. Creating default hoodie artifact...")
        garment = GarmentArtifact(
            artifact_id="hoodie_baggy_pants",
            name="Heavyweight Oversized Hoodie with Baggy Cargo Pants",
            category="streetwear",
            color="Charcoal Gray and Washed Olive",
            fabric="Heavyweight 450gsm Cotton Fleece & Rigid Cotton Twill",
            silhouette="Extremely oversized, dropped shoulders, relaxed wide-leg puddle hem",
        )

    # Execute Pipeline
    pipeline = FashionAIPipeline(output_dir=OUTPUT_DIR, mock_mode=args.mock)
    results = pipeline.run(profile=profile, garment_artifact=garment)
    recs = results['stylist_recommendations']

    print("\n" + "=" * 70)
    print("           FASHION AI PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    print(f"\n1. EXTRACTED BODY FEATURES (Saved: {results['body_features_file']}):")
    print(f"   - Body Shape: {results['body_features'].body_shape.upper()}")
    print(f"   - Color Season: {results['body_features'].color_season.upper()}")
    print(f"   - Undertone: {results['body_features'].tone.undertone.capitalize()} (Contrast: {results['body_features'].tone.contrast_level})")
    print(f"   - Summary: {results['body_features'].overall_summary}")

    print(f"\n2. BODY STRUCTURE RECOMMENDATIONS (DO's & DONT's):")
    print(f"   [Rationale]: {recs.body_dos_and_donts.proportional_rationale}")
    print("   [DO's]:")
    for item in recs.body_dos_and_donts.dos:
        print(f"     + {item}")
    print("   [DONT's]:")
    for item in recs.body_dos_and_donts.donts:
        print(f"     - {item}")

    print(f"\n3. SKIN ANALYSIS RECOMMENDATIONS (DO's & DONT's):")
    print(f"   [Rationale]: {recs.skin_dos_and_donts.chromatic_rationale}")
    print("   [DO's]:")
    for item in recs.skin_dos_and_donts.dos:
        print(f"     + {item}")
    print("   [DONT's]:")
    for item in recs.skin_dos_and_donts.donts:
        print(f"     - {item}")

    if results["tryon_verdict"]:
        verdict = results["tryon_verdict"]
        print(f"\n4. VIRTUAL TRY-ON VERDICT FOR ARTIFACT '{garment.name}':")
        print(f"   - Verdict: {verdict.verdict.upper()} (Score: {verdict.score_out_of_10}/10)")
        print(f"   - Headline: {verdict.verdict_headline}")
        print(f"   - Detailed Rationale: {verdict.detailed_rationale}")
        print(f"   - Video Try-on Voiceover / Subtitle Script:")
        print(f"     \"{verdict.tryon_video_subtitle_script}\"")
        print(f"   - Preview Image: {results['preview_image']}")
        print(f"   - Subtitle File (.srt): {results['subtitles_file']}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
