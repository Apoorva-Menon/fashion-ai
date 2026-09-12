import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

def resolve_image_path(path_str: Optional[str], search_dir: Path) -> Optional[str]:
    """Resolves an image path from absolute, CWD-relative, or search_dir-relative."""
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return str(p)
    
    cwd_p = Path.cwd() / p
    if cwd_p.exists():
        return str(cwd_p)
    
    search_p = search_dir / p
    if search_p.exists():
        return str(search_p)
    
    logger.warning(f"Could not locate image file: {path_str}")
    return None

def auto_discover_perspective_images(
    inputs_dir: Path,
    front: Optional[str] = None,
    side: Optional[str] = None,
    back: Optional[str] = None,
    angled: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """
    Intelligently finds and binds multi-perspective images from the inputs folder
    or user overrides.
    """
    resolved: Dict[str, Optional[str]] = {
        "front": resolve_image_path(front, inputs_dir),
        "side": resolve_image_path(side, inputs_dir),
        "back": resolve_image_path(back, inputs_dir),
        "angled": resolve_image_path(angled, inputs_dir),
    }

    if not inputs_dir.exists():
        return resolved

    # Scan all valid image files in inputs_dir sorted by modification time (newest first)
    all_images = [
        f for f in inputs_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    all_images.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # Helper to check if a file matches a perspective keyword
    def find_matching_file(keyword: str, exclude_names: list) -> Optional[Path]:
        # Prioritize files with '1' or newer versions first
        for img in all_images:
            name_lower = img.name.lower()
            if keyword in name_lower and img.name not in exclude_names:
                return img
        return None

    assigned = []

    # 1. Front discovery
    if not resolved["front"]:
        matched = find_matching_file("front", assigned)
        if matched:
            resolved["front"] = str(matched)
            assigned.append(matched.name)

    # 2. Side discovery
    if not resolved["side"]:
        matched = find_matching_file("side", assigned)
        if matched:
            resolved["side"] = str(matched)
            assigned.append(matched.name)

    # 3. Back discovery
    if not resolved["back"]:
        matched = find_matching_file("back", assigned) or find_matching_file("rear", assigned)
        if matched:
            resolved["back"] = str(matched)
            assigned.append(matched.name)

    # 4. Angled discovery
    if not resolved["angled"]:
        matched = find_matching_file("angle", assigned) or find_matching_file("45", assigned)
        if matched:
            resolved["angled"] = str(matched)
            assigned.append(matched.name)

    # Fallback: if front is still missing, pick first available image
    if not resolved["front"] and all_images:
        resolved["front"] = str(all_images[0])

    print("\n--- Image Auto-Discovery Summary ---")
    for key, p in resolved.items():
        if p:
            print(f"  [{key.upper():<6}]: {p}")
        else:
            print(f"  [{key.upper():<6}]: (None)")
    print("------------------------------------\n")

    return resolved
