import json
import logging
from typing import Optional, List
from fashion_ai.config import GEMINI_API_KEY, STYLIST_MODEL, MOCK_MODE
from fashion_ai.schemas.body_features import BodyFeatures
from fashion_ai.schemas.stylist_recommendations import (
    StylistRecommendations,
    BodyDosAndDonts,
    SkinAnalysisDosAndDonts,
    CategoryGuidance,
    SubtitleLine,
)
from fashion_ai.agents.skill_manager import SkillManager
from fashion_ai.prompts.stylist import STYLIST_SYSTEM_PROMPT, STYLIST_USER_PROMPT

logger = logging.getLogger(__name__)

class StylistAgent:
    """Skill-powered personal stylist recommending best vs avoid lists and video subtitles."""

    def __init__(
        self,
        model_name: str = STYLIST_MODEL,
        api_key: Optional[str] = None,
        mock_mode: bool = MOCK_MODE,
        skill_manager: Optional[SkillManager] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or GEMINI_API_KEY
        self.mock_mode = mock_mode
        self.skill_manager = skill_manager or SkillManager()
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

    def generate_recommendations(self, features: BodyFeatures) -> StylistRecommendations:
        """Loads specific body/color skills and produces verified recommendations with DOs and DONTs."""
        # 1. Dynamically load skills for the identified shape & season
        skill_text, activated_skills = self.skill_manager.load_skills(
            body_shape=features.body_shape,
            color_season=features.color_season,
        )

        if self.mock_mode or not self._client:
            return self._mock_recommend(features, activated_skills)

        from google import genai
        from google.genai import types

        prompt = STYLIST_USER_PROMPT.format(
            body_features_json=features.model_dump_json(indent=2),
            skill_content=skill_text,
        )

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=[prompt],
            config=types.GenerateContentConfig(
                system_instruction=STYLIST_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=StylistRecommendations,
                temperature=0.3,
            ),
        )

        recommendations = StylistRecommendations.model_validate_json(response.text)
        recommendations.activated_skills = activated_skills
        return recommendations

    def _mock_recommend(self, features: BodyFeatures, activated_skills: List[str]) -> StylistRecommendations:
        """Deterministic mock recommendations with granular body & skin DOs and DONTs."""
        shape = features.body_shape
        season = features.color_season
        undertone = features.tone.undertone

        # Specific Body DOs & DONTs based on silhouette
        if shape == "inverted_triangle":
            body_dos = [
                "DO wear deep V-necks, raglan sleeves, and scoop necklines to visually soften shoulder width.",
                "DO choose wide-leg trousers, cargo pockets, and A-line skirts to build lower-body volume and balance.",
                "DO pick unstructured, soft-draping fabrics on top (viscose, fine knits, silk) with structured fabrics on bottoms.",
                "DO use single-breasted coats with clean vertical lapels to elongate the upper torso.",
            ]
            body_donts = [
                "DON'T wear heavy shoulder pads, epaulets, or prominent puff sleeves that broaden the shoulder line.",
                "DON'T pair skinny tight jeans with oversized boxy tops (creates an unbalanced top-heavy ice-cream cone look).",
                "DON'T choose boatnecks or horizontal stripe sweaters across the upper chest.",
                "DON'T wear shapeless drop-shoulder heavyweight hoodies that obscure your vertical line.",
            ]
            proportional_rationale = (
                f"Your {shape} geometry features broader upper shoulders relative to hips. Styling DOs "
                f"draw visual interest downward and add architectural fullness to the hips, while DONTs "
                f"prevent unwanted upper bulk."
            )
        elif shape == "hourglass":
            body_dos = [
                "DO accentuate your natural waist with wrap tops, peplum cuts, and medium-to-wide belts.",
                "DO wear fit-and-flare dresses, high-waisted tailored trousers, and open sweetheart or scoop necklines.",
                "DO choose medium-weight fluid fabrics that hug contours without bunching.",
            ]
            body_donts = [
                "DON'T wear shapeless sack dresses or straight boxy tunics that hide waist indentation.",
                "DON'T wear heavy, stiff fabrics (e.g. rigid thick fleece) that mask your natural symmetry.",
                "DON'T use drop-waist silhouettes that disrupt hip-to-waist proportions.",
            ]
            proportional_rationale = (
                "Your bust and hips are balanced with a well-defined waist. DOs highlight your waist as the focal "
                "anchor, while DONTs eliminate shapeless boxiness that makes the frame look heavier."
            )
        else:
            body_dos = [
                "DO utilize structured tailoring and vertical seams to create clean lengthening lines.",
                "DO balance volume—pair relaxed silhouettes with fitted contrasting pieces.",
                "DO wear open necklines (V-neck, soft collar) to draw focus to collarbones and face.",
            ]
            body_donts = [
                "DON'T wear head-to-toe oversized shapeless garments with no focal point.",
                "DON'T wear stiff, unyielding fabrics that do not follow natural body motion.",
            ]
            proportional_rationale = f"Tailored specifically for {shape} proportions to maintain visual equilibrium."

        # Specific Skin Analysis DOs & DONTs based on color season & undertone
        if "autumn" in season:
            skin_dos = [
                "DO prioritize warm, grounded earth neutrals: rich espresso brown, deep camel, warm olive green, and dark cream.",
                "DO add vibrant warm accent pops: terracotta, burnt orange, rust red, and forest green.",
                "DO select warm jewelry metals: brushed yellow gold, antique bronze, and warm copper.",
                "DO choose matte, rich textures (velvet, suede, corduroy, raw silk) that absorb and enrich warm light.",
            ]
            skin_donts = [
                "DON'T wear stark, optical high-glare white (casts an unflattering gray pallor on warm undertones).",
                "DON'T wear icy, frosty pastels (pale baby pink, icy cyan, frosted lavender) which wash out warmth.",
                "DON'T wear pure jet black near the face without a warm scarf or jewelry break.",
                "DON'T wear high-shine cool chrome or cool silver metals that clash with golden skin tones.",
            ]
            chromatic_rationale = (
                f"With a {undertone} undertone and {features.tone.contrast_level} contrast in the {season} palette, "
                f"your complexion is illuminated by rich, warm, golden-based pigments and drained by cool, chalky tones."
            )
        else:
            skin_dos = [
                "DO wear clear, harmonious tones matching your seasonal palette to illuminate your complexion.",
                "DO choose metals and jewelry finishes that mirror your natural undertone temperature.",
            ]
            skin_donts = [
                "DON'T wear muddied or clashing color temperatures near your face.",
                "DON'T wear washed-out neons that overpower your natural contrast.",
            ]
            chromatic_rationale = f"Harmonized for {season} color season with {undertone} undertones."

        return StylistRecommendations(
            body_type_identified=shape,
            color_season_identified=season,
            activated_skills=activated_skills,
            body_dos_and_donts=BodyDosAndDonts(
                dos=body_dos,
                donts=body_donts,
                proportional_rationale=proportional_rationale,
            ),
            skin_dos_and_donts=SkinAnalysisDosAndDonts(
                dos=skin_dos,
                donts=skin_donts,
                chromatic_rationale=chromatic_rationale,
            ),
            what_suits_best=CategoryGuidance(
                fabrics=[
                    "Medium-weight structured wool for clean architecture",
                    "Matte silk and fluid viscose for effortless draping",
                    "Fine-gauge ribbed knitwear and cotton twill",
                ],
                silhouettes_and_cuts=[
                    "Clean open V-neck and scoop necklines to lengthen vertical line",
                    "Tailored straight or subtle wide-leg trousers to balance the frame",
                    "Single-breasted blazers hitting at the mid-hip with natural shoulder slope",
                ],
                colors_and_palettes=[
                    "Rich espresso brown",
                    "Deep terracotta & burnt orange accents",
                    "Warm olive green and dark camel neutrals",
                ],
            ),
            what_to_avoid=CategoryGuidance(
                fabrics=[
                    "Heavy, stiff, oversized fleece that adds bulky boxiness",
                    "Clingy thin synthetics that bunch at uneven seams",
                    "Stark shiny synthetics with cool reflective glare",
                ],
                silhouettes_and_cuts=[
                    "Extreme shapeless baggy hoodies without waist or shoulder definition",
                    "Drop-waist sack silhouettes that shorten leg proportions",
                    "Puff shoulder embellishments or horizontal chest striping",
                ],
                colors_and_palettes=[
                    "Icy pale baby pink and frosty cyan",
                    "Stark optical high-glare white",
                    "Harsh neon magenta",
                ],
            ),
            key_rules=[
                f"Rule 1: Always maintain balance for your {shape} geometry by avoiding unanchored upper-body volume.",
                f"Rule 2: Emphasize warmth and earthy saturation to compliment your {season} complexion.",
                "Rule 3: Pair volume with structure—never wear oversized tops with shapeless bottoms simultaneously.",
            ],
            video_subtitles=[
                SubtitleLine(
                    sequence=1,
                    timestamp="00:00 - 00:04",
                    narration=f"Welcome to your bespoke style breakdown! Today we're analyzing your {shape} silhouette and {season} palette.",
                ),
                SubtitleLine(
                    sequence=2,
                    timestamp="00:04 - 00:09",
                    narration="Notice how your natural proportions thrive on structured vertical lines rather than unanchored bulk.",
                ),
                SubtitleLine(
                    sequence=3,
                    timestamp="00:09 - 00:15",
                    narration="When choosing fabrics, lean into matte wools and rich earth tones like olive and terracotta, while avoiding stiff boxy fleece.",
                ),
                SubtitleLine(
                    sequence=4,
                    timestamp="00:15 - 00:20",
                    narration="Let's test our first piece in the virtual try-on room to see these rules in action!",
                ),
            ],
        )
