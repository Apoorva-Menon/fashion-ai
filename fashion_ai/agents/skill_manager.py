import logging
from pathlib import Path
from typing import Dict, List, Tuple
from fashion_ai.config import BODY_TYPE_SKILLS_DIR, COLOR_SEASON_SKILLS_DIR

logger = logging.getLogger(__name__)

class SkillManager:
    """Dynamically loads and manages modular domain knowledge markdown skills."""

    def __init__(self, body_skills_dir: Path = BODY_TYPE_SKILLS_DIR, color_skills_dir: Path = COLOR_SEASON_SKILLS_DIR):
        self.body_skills_dir = body_skills_dir
        self.color_skills_dir = color_skills_dir

    def resolve_body_type_skill(self, body_shape: str) -> Path:
        """Finds the skill file matching the body shape."""
        normalized = body_shape.lower().strip().replace(" ", "_").replace("-", "_")
        target = self.body_skills_dir / f"{normalized}.md"
        if not target.exists():
            # Fallback or alias handling
            aliases = {
                "athletic": "rectangle.md",
                "triangle": "pear.md",
                "round": "apple.md",
                "column": "rectangle.md",
            }
            if normalized in aliases and (self.body_skills_dir / aliases[normalized]).exists():
                return self.body_skills_dir / aliases[normalized]
            logger.warning(f"Skill file not found for body shape: {body_shape}, using fallback.")
            # Default to first available
            available = list(self.body_skills_dir.glob("*.md"))
            if available:
                return available[0]
        return target

    def resolve_color_season_skill(self, color_season: str) -> Path:
        """Finds the skill file matching the color season."""
        normalized = color_season.lower().strip().replace(" ", "_").replace("-", "_")
        target = self.color_skills_dir / f"{normalized}.md"
        if not target.exists():
            # Map neutral to autumn or closest
            if "autumn" in normalized:
                target = self.color_skills_dir / "deep_autumn.md"
            elif "winter" in normalized:
                target = self.color_skills_dir / "bright_winter.md"
            elif "summer" in normalized:
                target = self.color_skills_dir / "cool_summer.md"
            elif "spring" in normalized:
                target = self.color_skills_dir / "warm_spring.md"
            else:
                available = list(self.color_skills_dir.glob("*.md"))
                target = available[0] if available else target
        return target

    def load_skills(self, body_shape: str, color_season: str) -> Tuple[str, List[str]]:
        """Loads and combines the markdown text of relevant body & color skills."""
        body_skill_path = self.resolve_body_type_skill(body_shape)
        color_skill_path = self.resolve_color_season_skill(color_season)

        activated_skills = []
        combined_text = []

        if body_skill_path.exists():
            activated_skills.append(body_skill_path.name)
            combined_text.append(f"=== ACTIVATED KNOWLEDGE: {body_skill_path.name} ===\n" + body_skill_path.read_text(encoding="utf-8"))
        else:
            logger.warning(f"Body skill path does not exist: {body_skill_path}")

        if color_skill_path.exists():
            activated_skills.append(color_skill_path.name)
            combined_text.append(f"=== ACTIVATED KNOWLEDGE: {color_skill_path.name} ===\n" + color_skill_path.read_text(encoding="utf-8"))
        else:
            logger.warning(f"Color skill path does not exist: {color_skill_path}")

        return "\n\n".join(combined_text), activated_skills
