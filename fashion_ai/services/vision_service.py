import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class VisionService:
    """Wrapper for Google Cloud Vision API with offline mock fallback."""

    def __init__(self, mock_mode: bool = False):
        self.mock_mode = mock_mode
        self._client = None
        if not self.mock_mode:
            try:
                from google.cloud import vision
                self._client = vision.ImageAnnotatorClient()
            except Exception as e:
                logger.warning(f"Google Cloud Vision client could not initialize ({e}). Falling back to mock mode.")
                self.mock_mode = True

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Runs face detection, dominant color extraction, and object localization."""
        if self.mock_mode or not self._client:
            return self._mock_analysis(image_path)

        from google.cloud import vision

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        content = path.read_bytes()
        image = vision.Image(content=content)

        # Execute multi-feature detection
        features = [
            vision.Feature(type_=vision.Feature.Type.FACE_DETECTION),
            vision.Feature(type_=vision.Feature.Type.IMAGE_PROPERTIES),
            vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION),
            vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION),
        ]
        request = vision.AnnotateImageRequest(image=image, features=features)
        response = self._client.annotate_image(request)

        # 1. Dominant colors
        dominant_colors = []
        if response.image_properties_annotation:
            for color_info in response.image_properties_annotation.dominant_colors.colors[:5]:
                c = color_info.color
                r, g, b = int(c.red), int(c.green), int(c.blue)
                hex_val = f"#{r:02x}{g:02x}{b:02x}"
                dominant_colors.append({
                    "hex": hex_val,
                    "rgb": [r, g, b],
                    "score": round(color_info.score, 3),
                    "pixel_fraction": round(color_info.pixel_fraction, 3),
                })

        # 2. Face geometry
        face_data = []
        for face in response.face_annotations:
            face_data.append({
                "roll_angle": round(face.roll_angle, 2),
                "pan_angle": round(face.pan_angle, 2),
                "tilt_angle": round(face.tilt_angle, 2),
                "detection_confidence": round(face.detection_confidence, 2),
            })

        # 3. Localized apparel / objects
        objects = []
        for obj in response.localized_object_annotations:
            objects.append({
                "name": obj.name,
                "score": round(obj.score, 2),
            })

        return {
            "dominant_colors": dominant_colors,
            "face_annotations": face_data,
            "detected_objects": objects,
            "labels": [lbl.description for lbl in response.label_annotations[:8]],
        }

    def _mock_analysis(self, image_path: str) -> Dict[str, Any]:
        """Provides realistic mock visual cues for local testing."""
        return {
            "dominant_colors": [
                {"hex": "#d4a373", "rgb": [212, 163, 115], "score": 0.45, "pixel_fraction": 0.28},  # warm skin
                {"hex": "#3d2b1f", "rgb": [61, 43, 31], "score": 0.32, "pixel_fraction": 0.22},     # dark brown hair
                {"hex": "#2b2d42", "rgb": [43, 45, 66], "score": 0.18, "pixel_fraction": 0.15},     # navy apparel
            ],
            "face_annotations": [
                {"roll_angle": 1.2, "pan_angle": 0.5, "tilt_angle": -0.8, "detection_confidence": 0.98}
            ],
            "detected_objects": [
                {"name": "Person", "score": 0.96},
                {"name": "Outerwear", "score": 0.88},
                {"name": "Pants", "score": 0.84}
            ],
            "labels": ["Clothing", "Outerwear", "Standing", "Trousers", "Street fashion", "Denim"],
            "is_mock": True
        }
