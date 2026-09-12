from typing import List
from pydantic import BaseModel, Field

class CategoryGuidance(BaseModel):
    fabrics: List[str] = Field(..., description="Recommended or avoided fabric types and textures")
    silhouettes_and_cuts: List[str] = Field(..., description="Garment shapes, cuts, necklines, hemlines")
    colors_and_palettes: List[str] = Field(..., description="Specific color names or palettes")

class BodyDosAndDonts(BaseModel):
    dos: List[str] = Field(
        ...,
        description="Specific cuts, silhouettes, necklines, hemlines, and tailoring techniques that flatter this body type and proportions",
    )
    donts: List[str] = Field(
        ...,
        description="Specific cuts, volumes, and styles that disrupt, shorten, or unflatteringly distort this body silhouette",
    )
    proportional_rationale: str = Field(
        ...,
        description="Detailed anatomical rationale explaining how these DOs and DONTs balance the user's specific torso, shoulders, hips, and vertical line",
    )

class SkinAnalysisDosAndDonts(BaseModel):
    dos: List[str] = Field(
        ...,
        description="Specific core flattering colors, seasonal palettes, jewelry metals, and fabric finishes that illuminate this skin tone",
    )
    donts: List[str] = Field(
        ...,
        description="Specific colors, harsh stark shades, or finishes that wash out, drain, or clash with this skin undertone and contrast level",
    )
    chromatic_rationale: str = Field(
        ...,
        description="Detailed colorimetric rationale explaining how these DOs and DONTs enhance the user's undertone, overtone, and contrast level",
    )

class SubtitleLine(BaseModel):
    sequence: int = Field(..., description="Display order")
    timestamp: str = Field(..., description="Estimated timestamp (e.g. '00:00 - 00:05')")
    narration: str = Field(..., description="Narration voiceover / subtitle text explaining the styling rule")

class StylistRecommendations(BaseModel):
    body_type_identified: str = Field(..., description="Matched body shape")
    color_season_identified: str = Field(..., description="Matched color season")
    activated_skills: List[str] = Field(..., description="List of knowledge skill files loaded")
    body_dos_and_donts: BodyDosAndDonts = Field(..., description="Dedicated DOs and DONTs grounded in structural body features")
    skin_dos_and_donts: SkinAnalysisDosAndDonts = Field(..., description="Dedicated DOs and DONTs grounded in skin tone & color analysis")
    what_suits_best: CategoryGuidance = Field(..., description="Aggregated items, cuts, fabrics, and colors that flatter the user")
    what_to_avoid: CategoryGuidance = Field(..., description="Aggregated items, cuts, fabrics, and colors to steer clear of")
    key_rules: List[str] = Field(..., description="High-level rules of thumb for this user")
    video_subtitles: List[SubtitleLine] = Field(..., description="Narration script for styling video / presentation")
