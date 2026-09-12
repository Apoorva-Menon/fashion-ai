import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root if present
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# API Keys & Credentials
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

# Vonage Video API Configuration
VONAGE_APPLICATION_ID = os.getenv("VONAGE_APPLICATION_ID", "")
VONAGE_PRIVATE_KEY_PATH = os.getenv("VONAGE_PRIVATE_KEY_PATH", "")
VONAGE_PRIVATE_KEY = os.getenv("VONAGE_PRIVATE_KEY", "")

# Directory Paths
DATA_DIR = PROJECT_ROOT / "data"
INPUTS_DIR = DATA_DIR / "inputs"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
OUTPUT_DIR = Path(os.getenv("FASHION_OUTPUT_DIR", DATA_DIR / "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SKILLS_DIR = PROJECT_ROOT / "skills"
BODY_TYPE_SKILLS_DIR = SKILLS_DIR / "body_types"
COLOR_SEASON_SKILLS_DIR = SKILLS_DIR / "color_seasons"

# Model Selection (Current Google GenAI SDK standards)
FEATURE_EXTRACTOR_MODEL = os.getenv("FASHION_FEATURE_MODEL", "gemini-2.5-flash")
STYLIST_MODEL = os.getenv("FASHION_STYLIST_MODEL", "gemini-2.5-flash")
TRYON_MODEL = os.getenv("FASHION_TRYON_MODEL", "gemini-2.5-flash")

# Execution Mode
MOCK_MODE = os.getenv("FASHION_MOCK_MODE", "false").lower() in ("true", "1", "yes")
