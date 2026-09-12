import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class VisionService:
    """Wrapper for Google Cloud Vision API with support for file paths, in-memory bytes, and live video frames."""

    def __init__(self, mock_mode: bool = False):
        self.mock_mode = mock_mode
        self._client = None
        if not self.mock_mode:
            try:
                from google.cloud import vision
                self._client = vision.ImageAnnotatorClient()
                logger.info("Google Cloud Vision API client initialized successfully.")
            except Exception as e:
                logger.warning(f"Google Cloud Vision client could not initialize ({e}). Falling back to mock mode.")
                self.mock_mode = True

    def analyze_image(
        self,
        image_path: Optional[Union[str, Path]] = None,
        image_bytes: Optional[bytes] = None,
        frame: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Runs Vision API feature detection on:
          1. image_path (Path on disk)
          2. image_bytes (Raw bytes)
          3. frame (OpenCV numpy frame captured directly from live video)
        """
        if self.mock_mode or not self._client:
            return self._mock_analysis(image_path=image_path, frame=frame)

        from google.cloud import vision

        content = None
        if frame is not None:
            try:
                import cv2
                success, encoded_img = cv2.imencode(".jpg", frame)
                if not success:
                    raise ValueError("Failed to encode OpenCV video frame to JPEG.")
                content = encoded_img.tobytes()
            except ImportError:
                raise RuntimeError("OpenCV (cv2) is required to analyze raw video frames.")
        elif image_bytes is not None:
            content = image_bytes
        elif image_path is not None:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image not found at {image_path}")
            content = path.read_bytes()
        else:
            raise ValueError("Must provide either image_path, image_bytes, or frame.")

        image = vision.Image(content=content)

        # Multi-feature detection for fashion analysis
        features = [
            vision.Feature(type_=vision.Feature.Type.FACE_DETECTION),
            vision.Feature(type_=vision.Feature.Type.IMAGE_PROPERTIES),
            vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION),
            vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION),
        ]
        request = vision.AnnotateImageRequest(image=image, features=features)
        response = self._client.annotate_image(request)

        # 1. Dominant colors (for skin tone, hair, and apparel)
        dominant_colors = []
        if response.image_properties_annotation:
            for color_info in response.image_properties_annotation.dominant_colors.colors[:6]:
                c = color_info.color
                r, g, b = int(c.red), int(c.green), int(c.blue)
                hex_val = f"#{r:02x}{g:02x}{b:02x}"
                dominant_colors.append({
                    "hex": hex_val,
                    "rgb": [r, g, b],
                    "score": round(color_info.score, 3),
                    "pixel_fraction": round(color_info.pixel_fraction, 3),
                })

        # 2. Face geometry (for posture, angle, lighting check)
        face_data = []
        for face in response.face_annotations:
            face_data.append({
                "roll_angle": round(face.roll_angle, 2),
                "pan_angle": round(face.pan_angle, 2),
                "tilt_angle": round(face.tilt_angle, 2),
                "detection_confidence": round(face.detection_confidence, 2),
                "under_exposed_likelihood": str(face.under_exposed_likelihood).split(".")[-1],
                "blurred_likelihood": str(face.blurred_likelihood).split(".")[-1],
            })

        # 3. Localized objects (person, top, pants, outerwear)
        objects = []
        for obj in response.localized_object_annotations:
            bbox = []
            for vertex in obj.bounding_poly.normalized_vertices:
                bbox.append({"x": round(vertex.x, 3), "y": round(vertex.y, 3)})
            objects.append({
                "name": obj.name,
                "score": round(obj.score, 2),
                "bounding_poly": bbox,
            })

        return {
            "dominant_colors": dominant_colors,
            "face_annotations": face_data,
            "detected_objects": objects,
            "labels": [lbl.description for lbl in response.label_annotations[:8]],
            "is_mock": False,
        }

    def analyze_frame(self, frame: Any, perspective: str = "front") -> Dict[str, Any]:
        """Direct helper to analyze a live camera video frame."""
        result = self.analyze_image(frame=frame)
        result["perspective"] = perspective
        return result

    def analyze_perspectives(self, perspectives: Dict[str, Optional[Union[str, Path, Any]]]) -> Dict[str, Any]:
        """
        Analyzes multiple perspectives (front, side, angled) captured from live video or files,
        aggregating visual signals into a unified multi-angle report.
        """
        perspective_results: Dict[str, Any] = {}
        all_colors: List[Dict[str, Any]] = []
        detected_apparel: set = set()

        for name, source in perspectives.items():
            if not source:
                continue

            try:
                if isinstance(source, (str, Path)):
                    if not Path(source).exists():
                        continue
                    analysis = self.analyze_image(image_path=str(source))
                else:
                    # In-memory video frame
                    analysis = self.analyze_image(frame=source)

                analysis["perspective"] = name
                perspective_results[name] = analysis

                for c in analysis.get("dominant_colors", []):
                    all_colors.append(c)
                for obj in analysis.get("detected_objects", []):
                    detected_apparel.add(obj.get("name"))

            except Exception as e:
                logger.warning(f"Vision API analysis failed for {name} perspective: {e}")

        # If no perspectives could be loaded, fallback to mock if in mock mode
        if not perspective_results and self.mock_mode:
            mock_res = self._mock_analysis()
            perspective_results["front"] = mock_res
            all_colors = mock_res["dominant_colors"]
            detected_apparel = {"Person", "Outerwear", "Pants"}

        # Sort aggregated colors by score
        sorted_colors = sorted(all_colors, key=lambda x: x.get("score", 0), reverse=True)[:8]

        return {
            "perspectives": perspective_results,
            "composite_dominant_colors": sorted_colors,
            "detected_objects_summary": list(detected_apparel),
            "total_perspectives_analyzed": len(perspective_results),
            "is_mock": self.mock_mode or not bool(self._client),
        }

    def _mock_analysis(self, image_path: Optional[Union[str, Path]] = None, frame: Optional[Any] = None) -> Dict[str, Any]:
        """Provides realistic mock visual cues for offline testing."""
        return {
            "dominant_colors": [
                {"hex": "#d4a373", "rgb": [212, 163, 115], "score": 0.48, "pixel_fraction": 0.28},  # skin
                {"hex": "#3d2b1f", "rgb": [61, 43, 31], "score": 0.35, "pixel_fraction": 0.22},     # hair
                {"hex": "#2b2d42", "rgb": [43, 45, 66], "score": 0.20, "pixel_fraction": 0.15},     # apparel
            ],
            "face_annotations": [
                {
                    "roll_angle": 0.8,
                    "pan_angle": 0.4,
                    "tilt_angle": -0.5,
                    "detection_confidence": 0.99,
                    "under_exposed_likelihood": "VERY_UNLIKELY",
                    "blurred_likelihood": "VERY_UNLIKELY",
                }
            ],
            "detected_objects": [
                {"name": "Person", "score": 0.97},
                {"name": "Top", "score": 0.91},
                {"name": "Pants", "score": 0.86},
            ],
            "labels": ["Clothing", "Outerwear", "Standing", "Trousers", "Street fashion", "Denim"],
            "is_mock": True,
        }
