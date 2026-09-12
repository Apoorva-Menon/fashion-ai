import logging
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class CameraCaptureService:
    """Captures multi-perspective portraits from a live camera feed using OpenCV."""

    def __init__(self, camera_index: int = 0, output_dir: Optional[Path] = None):
        self.camera_index = camera_index
        self.output_dir = output_dir or Path("data/inputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def capture_perspectives_interactive(self) -> Dict[str, str]:
        """
        Interactive webcam capture session.
        Press:
          'f' -> capture front perspective
          's' -> capture side perspective
          'a' -> capture angled perspective
          'q' -> finish and save session
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
        print("\n--- Live Camera Multi-Perspective Capture ---")
        print("Controls:")
        print("  [F] : Capture Front Portrait")
        print("  [S] : Capture Side Profile Portrait")
        print("  [A] : Capture Angled 45° Portrait")
        print("  [Q] : Finish and Proceed")

        timestamp = int(time.time())

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to grab frame from camera.")
                    break

                # Overlay status
                display_frame = frame.copy()
                status_text = f"Captured: {', '.join(captured_paths.keys()) or 'None'}"
                cv2.putText(display_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press F: Front | S: Side | A: Angled | Q: Done", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                cv2.imshow("Fashion AI - Perspective Capture", display_frame)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord('f'), ord('F')):
                    path = self.output_dir / f"front_{timestamp}.jpg"
                    cv2.imwrite(str(path), frame)
                    captured_paths["front"] = str(path)
                    print(f"Captured Front: {path}")

                elif key in (ord('s'), ord('S')):
                    path = self.output_dir / f"side_{timestamp}.jpg"
                    cv2.imwrite(str(path), frame)
                    captured_paths["side"] = str(path)
                    print(f"Captured Side: {path}")

                elif key in (ord('a'), ord('A')):
                    path = self.output_dir / f"angled_{timestamp}.jpg"
                    cv2.imwrite(str(path), frame)
                    captured_paths["angled"] = str(path)
                    print(f"Captured Angled: {path}")

                elif key in (ord('q'), ord('Q'), 27):  # 27 = ESC
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

        return captured_paths
