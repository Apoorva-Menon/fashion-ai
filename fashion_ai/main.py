import argparse
import json
import logging
import sys
from pathlib import Path

from fashion_ai.config import INPUTS_DIR, ARTIFACTS_DIR, OUTPUT_DIR, MOCK_MODE
from fashion_ai.schemas.user_profile import PerspectiveImages, UserMetrics, UserProfile
from fashion_ai.schemas.tryon import GarmentArtifact
from fashion_ai.services.camera import CameraCaptureService
from fashion_ai.services.vision_service import VisionService
from fashion_ai.services.vonage_video_service import VonageVideoService
from fashion_ai.services.image_loader import auto_discover_perspective_images
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
    parser.add_argument("--camera", action="store_true", help="Launch interactive camera capture with real-time Vision API analysis")

    # Vonage Video API options
    parser.add_argument("--vonage", action="store_true", help="Initialize a Vonage WebRTC video room & generate browser client")
    parser.add_argument("--vonage-video", type=str, help="Path to a recorded Vonage video stream/archive to extract frames from")

    # Image inputs
    parser.add_argument("--front", type=str, help="Path or filename to front full-body image")
    parser.add_argument("--side", type=str, help="Path or filename to side profile full-body image")
    parser.add_argument("--back", type=str, help="Path or filename to back profile full-body image")
    parser.add_argument("--angled", type=str, help="Path or filename to angled 45-degree full-body image")

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

    live_vision_signals = None

    # Handle Vonage Video Session Generation
    if args.vonage:
        logger.info("Initializing Vonage Video API WebRTC session...")
        vonage_svc = VonageVideoService(mock_mode=args.mock)
        session_info = vonage_svc.create_video_session(media_mode="routed")
        token = vonage_svc.generate_client_token(session_info["session_id"])
        html_client_path = vonage_svc.generate_webrtc_html_client(
            session_id=session_info["session_id"],
            token=token,
            output_file=OUTPUT_DIR / "vonage_video_client.html",
        )
        print("\n" + "=" * 70)
        print("         VONAGE VIDEO API WEBRTC LIVE SESSION READY")
        print("=" * 70)
        print(f"Session ID  : {session_info['session_id']}")
        print(f"Client Token: {token[:35]}...")
        print(f"Browser WebRTC Client: file://{html_client_path.resolve()}")
        print("Open the HTML file in Chrome/Safari to stream camera and capture frames!")
        print("=" * 70 + "\n")

    # Handle Vonage Recorded Video Stream Ingestion
    if args.vonage_video:
        logger.info(f"Extracting multi-perspective frames from Vonage video: {args.vonage_video}")
        vonage_svc = VonageVideoService(mock_mode=args.mock)
        extracted = vonage_svc.extract_frames_from_video(args.vonage_video)
        import cv2
        for angle, frame in extracted.items():
            out_p = INPUTS_DIR / f"vonage_{angle}.jpg"
            cv2.imwrite(str(out_p), frame)
            if angle == "front": args.front = str(out_p)
            elif angle == "side": args.side = str(out_p)
            elif angle == "angled": args.angled = str(out_p)
        logger.info(f"Extracted {len(extracted)} perspective frames from Vonage video.")

    # Auto-discover or resolve images from data/inputs
    images = auto_discover_perspective_images(
        inputs_dir=INPUTS_DIR,
        front=args.front,
        side=args.side,
        back=args.back,
        angled=args.angled,
    )

    # If local camera capture requested, update paths
    if args.camera:
        logger.info("Opening live webcam capture with Google Vision API integration...")
        vision_service = VisionService(mock_mode=args.mock)
        cam_service = CameraCaptureService(output_dir=INPUTS_DIR, vision_service=vision_service)
        captured_paths, live_vision_signals = cam_service.capture_perspectives_interactive()
        for k, v in captured_paths.items():
            images[k] = v

    if not images["front"]:
        logger.error("No front-facing image found. Please place an image in data/inputs or pass --front.")
        sys.exit(1)

    # Assemble UserProfile
    profile = UserProfile(
        profile_id="user_demo",
        images=PerspectiveImages(
            front_image_path=images["front"],
            side_image_path=images["side"],
            back_image_path=images["back"],
            angled_image_path=images["angled"],
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
    results = pipeline.run(
        profile=profile,
        garment_artifact=garment,
        precomputed_vision_signals=live_vision_signals,
    )
    recs = results['stylist_recommendations']

    print("\n" + "=" * 70)
    print("           FASHION AI PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    print(f"\n1. GOOGLE VISION API ANALYSIS (Saved: {results['vision_analysis_file']}):")
    v_signals = results.get("vision_signals", {})
    print(f"   - Perspectives Analyzed: {v_signals.get('total_perspectives_analyzed', len(v_signals.get('perspectives', {})))}")
    print(f"   - Detected Objects: {', '.join(v_signals.get('detected_objects_summary', ['Person', 'Apparel']))}")
    if "composite_dominant_colors" in v_signals:
        color_hexes = [c['hex'] for c in v_signals['composite_dominant_colors'][:4]]
        print(f"   - Sampled Dominant Palette: {', '.join(color_hexes)}")

    print(f"\n2. EXTRACTED BODY FEATURES (Saved: {results['body_features_file']}):")
    print(f"   - Body Shape: {results['body_features'].body_shape.upper()}")
    print(f"   - Color Season: {results['body_features'].color_season.upper()}")
    print(f"   - Undertone: {results['body_features'].tone.undertone.capitalize()} (Contrast: {results['body_features'].tone.contrast_level})")
    print(f"   - Summary: {results['body_features'].overall_summary}")

    print(f"\n3. BODY STRUCTURE RECOMMENDATIONS (DO's & DONT's):")
    print(f"   [Rationale]: {recs.body_dos_and_donts.proportional_rationale}")
    print("   [DO's]:")
    for item in recs.body_dos_and_donts.dos:
        print(f"     + {item}")
    print("   [DONT's]:")
    for item in recs.body_dos_and_donts.donts:
        print(f"     - {item}")

    print(f"\n4. SKIN ANALYSIS RECOMMENDATIONS (DO's & DONT's):")
    print(f"   [Rationale]: {recs.skin_dos_and_donts.chromatic_rationale}")
    print("   [DO's]:")
    for item in recs.skin_dos_and_donts.dos:
        print(f"     + {item}")
    print("   [DONT's]:")
    for item in recs.skin_dos_and_donts.donts:
        print(f"     - {item}")

    if results["tryon_verdict"]:
        verdict = results["tryon_verdict"]
        print(f"\n5. VIRTUAL TRY-ON VERDICT FOR ARTIFACT '{garment.name}':")
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
