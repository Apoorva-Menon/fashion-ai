# Fashion AI: Multi-Perspective Visual Styling & Virtual Try-On Pipeline

A modular Python framework combining Google Cloud Vision API, Google Gemini Multimodal models (`google-genai` SDK), and dynamic knowledge skills to analyze user body geometry, determine color seasons, deliver bespoke stylist recommendations (DOs & DON'Ts), and evaluate garment artifacts (e.g. *Oversized Hoodie with Baggy Pants*) for virtual try-on with video subtitles.

---

## Architecture Overview

```
                          ┌───────────────────────────┐
                          │   User Inputs:            │
                          │   - Multi-Perspective Imgs│
                          │     (Front, Side, Angled) │
                          │   - Height, Weight        │
                          │   - Reported Skin Tone    │
                          └─────────────┬─────────────┘
                                        │
                         ┌──────────────┴─────────────┐
                         │   Stage 1: Vision Analysis │
                         │   (Google Cloud Vision API)│
                         │   Dominant Colors & BBoxes │
                         └──────────────┬─────────────┘
                                        │
                         ┌──────────────┴─────────────┐
                         │ Stage 2: Feature Extractor │
                         │ (Gemini Multimodal LLM)    │
                         │ Output: body_features.json │
                         └──────────────┬─────────────┘
                                        │
                         ┌──────────────┴─────────────┐
                         │  Stage 3: Skill Dispatcher │
                         │  Loads Markdown Knowledge: │
                         │  - skills/body_types/*.md  │
                         │  - skills/color_seasons/*.md│
                         └──────────────┬─────────────┘
                                        │
                         ┌──────────────┴─────────────┐
                         │ Stage 4: Stylist Agent     │
                         │ Output:                    │
                         │ stylist_recommendations.json│
                         │ (DOs/DON'Ts + Subtitles)   │
                         └──────────────┬─────────────┘
                                        │
           ┌────────────────────────────┴───────────────────────────┐
           │                                                        │
┌──────────▼──────────────┐                              ┌──────────▼──────────────┐
│  Artifacts Ingestion    │                              │  Stage 5: Try-On Agent  │
│  data/artifacts/        │                              │  Evaluates Garment:     │
│  - hoodie_baggy_pants   ├─────────────────────────────►│  - Rating out of 10     │
│  - tailored_blazer      │                              │  - GOOD / NOT GOOD list │
└─────────────────────────┘                              │  - Try-on Video Subtitle│
                                                         └──────────┬──────────────┘
                                                                    │
                                                         ┌──────────▼──────────────┐
                                                         │ Visual Try-On Preview & │
                                                         │ Subtitles (.png, .srt)  │
                                                         └─────────────────────────┘
```

---

## Directory Structure

```
fashion-ai/
├── README.md                          # Documentation
├── pyproject.toml                     # Modern package & build settings
├── requirements.txt                   # Direct requirements
├── .env.example                       # API keys & model environment config
├── .gitignore                         # Output & cache ignore rules
├── data/
│   ├── inputs/                        # User perspective photos (front, side, angled)
│   ├── artifacts/                     # Garment artifacts (JSON specs + images)
│   └── output/                        # Generated JSONs, previews, .srt subtitles
├── skills/                            # Modular knowledge skills
│   ├── body_types/
│   │   ├── hourglass.md
│   │   ├── pear.md
│   │   ├── rectangle.md
│   │   ├── inverted_triangle.md
│   │   └── apple.md
│   └── color_seasons/
│       ├── warm_spring.md
│       ├── cool_summer.md
│       ├── deep_autumn.md
│       └── bright_winter.md
├── tests/
│   ├── test_schemas.py                # Schema validation tests
│   └── test_pipeline.py               # End-to-end pipeline tests
└── fashion_ai/
    ├── config.py                      # Model IDs, paths, env configuration
    ├── main.py                        # CLI entry point
    ├── pipeline.py                    # FashionAIPipeline orchestrator
    ├── schemas/
    │   ├── user_profile.py            # UserMetrics & PerspectiveImages
    │   ├── body_features.py           # BodyFeatures & Proportions
    │   ├── stylist_recommendations.py # StylistRecommendations & SubtitleLine
    │   └── tryon.py                   # GarmentArtifact & TryOnVerdict
    ├── services/
    │   ├── camera.py                  # OpenCV live camera feed capture
    │   ├── vision_service.py          # Google Cloud Vision API integration
    │   └── tryon_service.py           # Visual try-on preview & subtitle generator
    ├── agents/
    │   ├── skill_manager.py           # Dynamic markdown skill loader
    │   ├── feature_extractor_agent.py # Multimodal body structure & tone extractor
    │   ├── stylist_agent.py           # Skill-powered stylist recommender
    │   └── tryon_evaluator_agent.py   # Garment try-on evaluator
    └── prompts/
        ├── feature_extraction.py      # System/User prompts for feature analysis
        ├── stylist.py                 # System/User prompts for stylist advice
        └── tryon.py                   # System/User prompts for garment verdicts
```

---

## Quickstart

### 1. Installation

Using `uv` (recommended):
```bash
uv venv --python python3
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Environment Configuration

Copy `.env.example` to `.env` and set your credentials:
```bash
cp .env.example .env
```
- `GEMINI_API_KEY`: Obtain from [Google AI Studio](https://aistudio.google.com/app/api-keys).
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to your Google Cloud service account JSON (for Google Cloud Vision API).

### 3. Running the Pipeline

#### Offline / Mock Mode (No credentials required)
Test the entire pipeline immediately with sample profiles and pre-generated artifacts:
```bash
# Evaluate the Hoodie with Baggy Pants artifact (Classified as NOT GOOD)
python3 -m fashion_ai.main --mock --artifact data/artifacts/hoodie_with_baggy_pants.json

# Evaluate the Tailored Blazer artifact (Classified as GOOD)
python3 -m fashion_ai.main --mock --artifact data/artifacts/tailored_blazer_look.json
```

#### With Custom User Metrics
```bash
python3 -m fashion_ai.main --mock \
  --front data/inputs/sample_front.jpg \
  --side data/inputs/sample_side.jpg \
  --height 182.0 \
  --weight 78.0 \
  --skin-tone "Warm olive" \
  --gender "masculine"
```

#### Live Webcam Multi-Perspective Capture
To capture front and side perspectives directly from your webcam:
```bash
python3 -m fashion_ai.main --camera --height 172.0 --weight 66.0
```
Interactive controls in camera window:
- `[F]`: Capture Front full portrait
- `[S]`: Capture Side profile
- `[A]`: Capture Angled 45° view
- `[Q]`: Proceed to analysis

---

## Outputs Generated

All artifacts and evaluation files are saved to `data/output/`:
- `<profile_id>_body_features.json`: Full morphological & colorimetric analysis.
- `<profile_id>_stylist_recommendations.json`: Best vs Avoid cuts, fabrics, colors, and educational subtitles.
- `<profile_id>_tryon_<artifact_id>.json`: Compatibility score, verdict ("good", "not_good", "conditional"), and styling tweaks.
- `tryon_preview_<artifact_id>.png`: Visual side-by-side card with verdict banner.
- `tryon_narration_<artifact_id>.srt`: Subtitle file ready for video playback and narration sync.

---

## Running Tests

```bash
pytest -v
```
