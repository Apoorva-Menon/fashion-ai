import logging
from pathlib import Path
from typing import Optional
from PIL import Image, ImageDraw, ImageFont
from fashion_ai.config import OUTPUT_DIR
from fashion_ai.schemas.tryon import TryOnVerdict, GarmentArtifact

logger = logging.getLogger(__name__)

class TryOnService:
    """Generates visual try-on comparison artifacts, preview cards, and subtitle files."""

    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_tryon_preview(
        self,
        user_image_path: Optional[str],
        artifact: GarmentArtifact,
        verdict: TryOnVerdict,
    ) -> str:
        """
        Creates a side-by-side visual comparison preview card:
        [User Profile Photo] + [Garment Artifact] with verdict badge and subtitle banner.
        """
        output_file = self.output_dir / f"tryon_preview_{artifact.artifact_id}.png"
        card_w, card_h = 1000, 600

        preview = Image.new("RGB", (card_w, card_h), color=(24, 24, 27))
        draw = ImageDraw.Draw(preview)

        # 1. User image panel
        user_img = None
        if user_image_path and Path(user_image_path).exists():
            try:
                user_img = Image.open(user_image_path).convert("RGB")
                user_img.thumbnail((440, 440))
                preview.paste(user_img, (40, 50))
            except Exception as e:
                logger.warning(f"Failed to paste user image: {e}")

        if not user_img:
            # Draw placeholder box
            draw.rectangle([(40, 50), (480, 490)], outline=(80, 80, 80), width=2, fill=(40, 40, 45))
            draw.text((160, 260), "User Profile Perspective", fill=(200, 200, 200))

        # 2. Garment artifact panel
        garment_img = None
        if artifact.image_path and Path(artifact.image_path).exists():
            try:
                garment_img = Image.open(artifact.image_path).convert("RGB")
                garment_img.thumbnail((440, 440))
                preview.paste(garment_img, (520, 50))
            except Exception as e:
                logger.warning(f"Failed to paste garment image: {e}")

        if not garment_img:
            draw.rectangle([(520, 50), (960, 490)], outline=(80, 80, 80), width=2, fill=(40, 40, 45))
            draw.text((600, 240), f"Artifact: {artifact.name[:25]}", fill=(220, 220, 220))
            draw.text((600, 270), f"Color: {artifact.color}", fill=(180, 180, 180))
            draw.text((600, 300), f"Fabric: {artifact.fabric}", fill=(180, 180, 180))

        # 3. Verdict badge at bottom
        badge_colors = {
            "good": (34, 197, 94),        # Green
            "not_good": (239, 68, 68),    # Red
            "conditional": (245, 158, 11) # Amber
        }
        badge_color = badge_colors.get(verdict.verdict, (156, 163, 175))
        badge_text = f"VERDICT: {verdict.verdict.upper()} (Score: {verdict.score_out_of_10}/10)"

        draw.rectangle([(40, 515), (960, 575)], fill=(30, 30, 35), outline=badge_color, width=2)
        draw.text((60, 525), badge_text, fill=badge_color)
        draw.text((60, 548), f"Headline: {verdict.verdict_headline[:100]}", fill=(240, 240, 240))

        preview.save(output_file)
        logger.info(f"Generated tryon preview card at {output_file}")
        return str(output_file)

    def export_subtitles(self, verdict: TryOnVerdict) -> str:
        """Exports the try-on voiceover/script as a standard .srt subtitle file."""
        srt_file = self.output_dir / f"tryon_narration_{verdict.artifact_id}.srt"
        content = (
            f"1\n"
            f"00:00:00,500 --> 00:00:06,000\n"
            f"{verdict.verdict_headline}\n\n"
            f"2\n"
            f"00:00:06,500 --> 00:00:15,000\n"
            f"{verdict.tryon_video_subtitle_script}\n"
        )
        srt_file.write_text(content, encoding="utf-8")
        return str(srt_file)
