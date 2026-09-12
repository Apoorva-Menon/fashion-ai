import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class VonageVideoService:
    """
    Integrates Vonage Video API (WebRTC) for remote live video streaming,
    session management, client token generation, and video frame extraction for Vision API.
    """

    def __init__(
        self,
        application_id: Optional[str] = None,
        private_key: Optional[str] = None,
        private_key_path: Optional[str] = None,
        mock_mode: bool = False,
    ):
        self.application_id = application_id or os.getenv("VONAGE_APPLICATION_ID", "")
        self.private_key = private_key or os.getenv("VONAGE_PRIVATE_KEY", "")
        self.private_key_path = private_key_path or os.getenv("VONAGE_PRIVATE_KEY_PATH", "")
        self.mock_mode = mock_mode
        self._video_client = None

        if not self.mock_mode and self.application_id:
            try:
                from vonage import Auth, Vonage
                from vonage_video import Video

                key_content = self.private_key
                if not key_content and self.private_key_path and Path(self.private_key_path).exists():
                    key_content = Path(self.private_key_path).read_text(encoding="utf-8")

                if key_content:
                    auth = Auth(application_id=self.application_id, private_key=key_content)
                    vonage_client = Vonage(auth=auth)
                    self._video_client = Video(vonage_client)
                    logger.info("Vonage Video API client initialized successfully.")
                else:
                    logger.warning("Vonage private key not provided. Running Vonage service in mock mode.")
                    self.mock_mode = True
            except Exception as e:
                logger.warning(f"Failed to initialize Vonage Video client ({e}). Defaulting to mock mode.")
                self.mock_mode = True
        else:
            self.mock_mode = True

    def create_video_session(self, media_mode: str = "routed") -> Dict[str, Any]:
        """Creates a WebRTC video session for live camera streaming."""
        if self.mock_mode or not self._video_client:
            mock_session_id = f"2_MX4xMDB-flN1biBTZXAgMTIgMTQ6MjU6MDAgUERUIDIwMjZ-MC44OTk3Njg4fg-{int(time.time())}"
            return {
                "session_id": mock_session_id,
                "media_mode": media_mode,
                "is_mock": True,
            }

        from vonage_video import SessionOptions, MediaMode
        mm = MediaMode.ROUTED if media_mode == "routed" else MediaMode.RELAYED
        session = self._video_client.create_session(SessionOptions(media_mode=mm))
        return {
            "session_id": session.session_id,
            "media_mode": media_mode,
            "is_mock": False,
        }

    def generate_client_token(
        self,
        session_id: str,
        role: str = "publisher",
        data: str = "fashion_ai_client",
        expire_time: Optional[int] = None,
    ) -> str:
        """Generates a secure WebRTC client token for browser/mobile video streaming."""
        if self.mock_mode or not self._video_client:
            return f"T1==c2Vzc2lvbl9pZD0ybW9ja190b2tlbl9mYXNoaW9uX2Fp_{int(time.time())}"

        from vonage_video import TokenOptions, TokenRole
        token_role = TokenRole.PUBLISHER if role == "publisher" else TokenRole.SUBSCRIBER
        opts = TokenOptions(role=token_role, data=data, expire_time=expire_time)
        return self._video_client.generate_client_token(session_id, opts)

    def extract_frames_from_video(
        self,
        video_path: str,
        target_timestamps_sec: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Extracts high-resolution video frames at specified timestamps (or uniform intervals)
        from a recorded Vonage video stream / archive.
        Returns a dict mapping timestamp/angle to raw BGR numpy frames.
        """
        import cv2

        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found at {video_path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file at {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps

        if not target_timestamps_sec:
            # Default to extracting Front (25% into video), Side (50%), Angled (75%)
            target_timestamps_sec = [
                duration_sec * 0.25,
                duration_sec * 0.50,
                duration_sec * 0.75,
            ]

        extracted_frames: Dict[str, Any] = {}
        labels = ["front", "side", "angled"]

        for idx, sec in enumerate(target_timestamps_sec):
            frame_no = int(sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_no, total_frames - 1))
            ret, frame = cap.read()
            if ret:
                label = labels[idx] if idx < len(labels) else f"timestamp_{sec:.1f}s"
                extracted_frames[label] = frame

        cap.release()
        return extracted_frames

    def generate_webrtc_html_client(
        self,
        session_id: str,
        token: str,
        output_file: Optional[Path] = None,
    ) -> Path:
        """
        Generates a standalone Vonage Video WebRTC HTML client that opens in any browser,
        streams the camera, and allows snapping front and side frames for the Fashion AI pipeline.
        """
        dest = output_file or Path("data/output/vonage_video_client.html")
        dest.parent.mkdir(parents=True, exist_ok=True)

        api_key = self.application_id or "VONAGE_API_KEY"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Fashion AI - Vonage Live Video Room</title>
    <script src="https://static.opentok.com/v2/js/opentok.min.js"></script>
    <style>
        body {{
            background: #121214;
            color: #f4f4f5;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 24px;
        }}
        h1 {{ margin-bottom: 8px; }}
        p {{ color: #a1a1aa; margin-top: 0; }}
        #video-container {{
            width: 720px;
            height: 480px;
            background: #27272a;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }}
        .controls {{
            margin-top: 20px;
            display: flex;
            gap: 12px;
        }}
        button {{
            background: #3b82f6;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: 0.2s;
        }}
        button:hover {{ background: #2563eb; }}
        .badge {{
            background: #22c55e;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <h1>Fashion AI: Vonage Video Live Session</h1>
    <p>Session ID: <code>{session_id[:20]}...</code> | Status: <span class="badge">Active</span></p>

    <div id="video-container"></div>

    <div class="controls">
        <button onclick="capturePerspective('front')">📸 Capture Front Perspective</button>
        <button onclick="capturePerspective('side')">📸 Capture Side Profile</button>
        <button onclick="capturePerspective('angled')">📸 Capture Angled 45°</button>
    </div>

    <script>
        const apiKey = "{api_key}";
        const sessionId = "{session_id}";
        const token = "{token}";

        const session = OT.initSession(apiKey, sessionId);
        const publisher = OT.initPublisher("video-container", {{
            insertMode: "append",
            width: "100%",
            height: "100%"
        }});

        session.connect(token, function(error) {{
            if (error) {{
                console.error("Failed to connect to Vonage session:", error);
            }} else {{
                session.publish(publisher);
            }}
        }});

        function capturePerspective(name) {{
            const imgData = publisher.getImgData();
            const link = document.createElement("a");
            link.download = name + "_" + Date.now() + ".png";
            link.href = "data:image/png;base64," + imgData;
            link.click();
            alert("Captured " + name.toUpperCase() + " perspective frame! Pass this to the Fashion AI pipeline.");
        }}
    </script>
</body>
</html>"""
        dest.write_text(html_content, encoding="utf-8")
        logger.info(f"Generated Vonage Video HTML client at {dest}")
        return dest
