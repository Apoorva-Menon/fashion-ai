import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class CameraCaptureService:
    """Captures multi-perspective portraits from live video feed with instant Vision API analysis."""

    def __init__(
        self,
        camera_index: int = 0,
        output_dir: Optional[Path] = None,
        vision_service: Optional[Any] = None,
    ):
        self.camera_index = camera_index
        self.output_dir = output_dir or Path("data/inputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vision_service = vision_service

    def capture_perspectives_interactive(self) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Interactive webcam capture session with real-time Vision API feature extraction.
        Press:
          'f' -> capture front perspective + run Vision API
          's' -> capture side perspective + run Vision API
          'a' -> capture angled perspective + run Vision API
          'q' -> finish and return paths and vision signals
        """
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV is not installed. Install with 'pip install opencv-python'.")
            raise RuntimeError("OpenCV (cv2) required for live camera capture.")

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera at index {self.camera_index}")

        captured_paths: Dict[str, str] = {}
        captured_vision_signals: Dict[str, Any] = {}

        print("\n" + "=" * 65)
        print("   LIVE CAMERA CAPTURE WITH GOOGLE VISION API INTEGRATION")
        print("=" * 65)
        print("Controls:")
        print("  [F] : Capture Front Portrait (+ Vision API analysis)")
        print("  [S] : Capture Side Profile Portrait (+ Vision API analysis)")
        print("  [A] : Capture Angled 45° Portrait (+ Vision API analysis)")
        print("  [Q] : Finish and Proceed with Pipeline")
        print("=" * 65 + "\n")

        timestamp = int(time.time())
        last_feedback = "Ready. Stand in frame."

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to grab frame from camera.")
                    break

                display_frame = frame.copy()
                status_text = f"Captured: {', '.join(captured_paths.keys()) or 'None'}"
                cv2.putText(display_frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press F: Front | S: Side | A: Angled | Q: Done", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(display_frame, f"Vision Status: {last_feedback[:50]}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)

                cv2.imshow("Fashion AI - Live Vision Capture", display_frame)
                key = cv2.waitKey(1) & 0xFF

                target_key = None
                if key in (ord('f'), ord('F')):
                    target_key = "front"
                elif key in (ord('s'), ord('S')):
                    target_key = "side"
                elif key in (ord('a'), ord('A')):
                    target_key = "angled"
                elif key in (ord('q'), ord('Q'), 27):  # 27 = ESC
                    break

                if target_key:
                    path = self.output_dir / f"{target_key}_{timestamp}.jpg"
                    cv2.imwrite(str(path), frame)
                    captured_paths[target_key] = str(path)
                    print(f"\n[CAMERA] Captured {target_key.upper()} perspective -> {path}")

                    # Run Vision API directly on the live captured frame
                    if self.vision_service:
                        print(f"[VISION API] Analyzing captured {target_key} live video frame...")
                        try:
                            vision_data = self.vision_service.analyze_frame(frame, perspective=target_key)
                            captured_vision_signals[target_key] = vision_data

                            # Extract quick metrics for user feedback
                            colors = [c['hex'] for c in vision_data.get('dominant_colors', [])[:3]]
                            person_objs = [o['name'] for o in vision_data.get('detected_objects', [])]
                            face_count = len(vision_data.get('face_annotations', []))

                            feedback_msg = f"{target_key.upper()}: Face detected={face_count > 0}, Objects={person_objs[:2]}, Colors={colors}"
                            print(f"[VISION API RESULT] {feedback_msg}")
                            last_feedback = f"{target_key}: Detected {', '.join(person_objs[:2]) or 'Person'}"
                        except Exception as e:
                            logger.warning(f"Live Vision API analysis failed: {e}")
                            last_feedback = f"{target_key}: Vision API error"
                    else:
                        last_feedback = f"{target_key} saved"

        finally:
            cap.release()
            cv2.destroyAllWindows()

        return captured_paths, captured_vision_signals
