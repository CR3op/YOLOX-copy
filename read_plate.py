#!/usr/bin/env python3
"""Read a license plate, compute a confidence score, and learn from human feedback.

Put plate photos in datasets/plates, or pass an image path inside this
project. Photos already sitting in datasets/ are read too. Reading uses
the local Windows OCR engine, OpenCV image enhancement, and a local
human-feedback memory file (datasets/plate_feedback.json) so everything
stays inside this folder and nothing is downloaded from outside.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Photos in datasets/plates are the intended drop folder. Top-level files in
# datasets/ are included so a picture placed there is still read.
PLATE_DIRS = [ROOT / "datasets" / "plates", ROOT / "datasets"]
FEEDBACK_PATH = ROOT / "datasets" / "plate_feedback.json"

# Names printed on US plates. Matched against OCR text to decide the issuing state.
STATE_NAMES = {
    "ALABAMA": "Alabama",
    "ALASKA": "Alaska",
    "ARIZONA": "Arizona",
    "ARKANSAS": "Arkansas",
    "CALIFORNIA": "California",
    "COLORADO": "Colorado",
    "CONNECTICUT": "Connecticut",
    "DELAWARE": "Delaware",
    "DISTRICT OF COLUMBIA": "District of Columbia",
    "FLORIDA": "Florida",
    "GEORGIA": "Georgia",
    "HAWAII": "Hawaii",
    "IDAHO": "Idaho",
    "ILLINOIS": "Illinois",
    "INDIANA": "Indiana",
    "IOWA": "Iowa",
    "KANSAS": "Kansas",
    "KENTUCKY": "Kentucky",
    "LOUISIANA": "Louisiana",
    "MAINE": "Maine",
    "MARYLAND": "Maryland",
    "MASSACHUSETTS": "Massachusetts",
    "MICHIGAN": "Michigan",
    "MINNESOTA": "Minnesota",
    "MISSISSIPPI": "Mississippi",
    "MISSOURI": "Missouri",
    "MONTANA": "Montana",
    "NEBRASKA": "Nebraska",
    "NEVADA": "Nevada",
    "NEW HAMPSHIRE": "New Hampshire",
    "NEW JERSEY": "New Jersey",
    "NEW MEXICO": "New Mexico",
    "NEW YORK": "New York",
    "NORTH CAROLINA": "North Carolina",
    "NORTH DAKOTA": "North Dakota",
    "OHIO": "Ohio",
    "OKLAHOMA": "Oklahoma",
    "OREGON": "Oregon",
    "PENNSYLVANIA": "Pennsylvania",
    "RHODE ISLAND": "Rhode Island",
    "SOUTH CAROLINA": "South Carolina",
    "SOUTH DAKOTA": "South Dakota",
    "TENNESSEE": "Tennessee",
    "TEXAS": "Texas",
    "UTAH": "Utah",
    "VERMONT": "Vermont",
    "VIRGINIA": "Virginia",
    "WASHINGTON": "Washington",
    "WEST VIRGINIA": "West Virginia",
    "WISCONSIN": "Wisconsin",
    "WYOMING": "Wyoming",
}

# Slogans that identify a state and appear as printed plate contents.
SLOGANS = {
    "THE LONE STAR STATE": ("Texas", "The Lone Star State"),
    "LONE STAR STATE": ("Texas", "The Lone Star State"),
    "THE SUNSHINE STATE": ("Florida", "The Sunshine State"),
    "THE GOLDEN STATE": ("California", "The Golden State"),
    "EMPIRE STATE": ("New York", "Empire State"),
    "THE GARDEN STATE": ("New Jersey", "The Garden State"),
    "GRAND CANYON STATE": ("Arizona", "Grand Canyon State"),
    "THE ALOHA STATE": ("Hawaii", "The Aloha State"),
    "THE LAST FRONTIER": ("Alaska", "The Last Frontier"),
    "THE NATURAL STATE": ("Arkansas", "The Natural State"),
    "LAND OF ENCHANTMENT": ("New Mexico", "Land of Enchantment"),
    "CENTENNIAL STATE": ("Colorado", "Centennial State"),
    "CONSTITUTION STATE": ("Connecticut", "Constitution State"),
    "THE FIRST STATE": ("Delaware", "The First State"),
    "PEACH STATE": ("Georgia", "Peach State"),
    "GEM STATE": ("Idaho", "Gem State"),
    "PRAIRIE STATE": ("Illinois", "Prairie State"),
    "HOOSIER STATE": ("Indiana", "Hoosier State"),
    "HAWKEYE STATE": ("Iowa", "Hawkeye State"),
    "SUNFLOWER STATE": ("Kansas", "Sunflower State"),
    "BLUEGRASS STATE": ("Kentucky", "Bluegrass State"),
    "PELICAN STATE": ("Louisiana", "Pelican State"),
    "PINE TREE STATE": ("Maine", "Pine Tree State"),
    "OLD LINE STATE": ("Maryland", "Old Line State"),
    "BAY STATE": ("Massachusetts", "Bay State"),
    "GREAT LAKES STATE": ("Michigan", "Great Lakes State"),
    "NORTH STAR STATE": ("Minnesota", "North Star State"),
    "10,000 LAKES": ("Minnesota", "10,000 Lakes"),
    "MAGNOLIA STATE": ("Mississippi", "Magnolia State"),
    "SHOW ME STATE": ("Missouri", "Show Me State"),
    "TREASURE STATE": ("Montana", "Treasure State"),
    "CORNHUSKER STATE": ("Nebraska", "Cornhusker State"),
    "SILVER STATE": ("Nevada", "Silver State"),
    "GRANITE STATE": ("New Hampshire", "Granite State"),
    "TAR HEEL STATE": ("North Carolina", "Tar Heel State"),
    "PEACE GARDEN STATE": ("North Dakota", "Peace Garden State"),
    "BUCKEYE STATE": ("Ohio", "Buckeye State"),
    "SOONER STATE": ("Oklahoma", "Sooner State"),
    "BEAVER STATE": ("Oregon", "Beaver State"),
    "KEYSTONE STATE": ("Pennsylvania", "Keystone State"),
    "OCEAN STATE": ("Rhode Island", "Ocean State"),
    "PALMETTO STATE": ("South Carolina", "Palmetto State"),
    "MOUNT RUSHMORE STATE": ("South Dakota", "Mount Rushmore State"),
    "VOLUNTEER STATE": ("Tennessee", "Volunteer State"),
    "BEEHIVE STATE": ("Utah", "Beehive State"),
    "GREEN MOUNTAIN STATE": ("Vermont", "Green Mountain State"),
    "OLD DOMINION": ("Virginia", "Old Dominion"),
    "EVERGREEN STATE": ("Washington", "Evergreen State"),
    "MOUNTAIN STATE": ("West Virginia", "Mountain State"),
    "BADGER STATE": ("Wisconsin", "Badger State"),
    "EQUALITY STATE": ("Wyoming", "Equality State"),
    "COWBOY STATE": ("Wyoming", "Cowboy State"),
}

# Words that describe the kind of plate, such as trailer or truck, not the number.
PLATE_TYPES = {
    "TRAILER": "Trailer",
    "TRUCK": "Truck",
    "PASSENGER": "Passenger",
    "COMMERCIAL": "Commercial",
    "MOTORCYCLE": "Motorcycle",
    "APPORTIONED": "Apportioned",
    "DISABLED": "Disabled",
    "FLEET": "Fleet",
    "EXEMPT": "Exempt",
    "DEALER": "Dealer",
}

PLATE_TYPE_PREFIXES = {
    "TRAIL": "Trailer",
    "PASSENG": "Passenger",
    "COMMERC": "Commercial",
    "MOTORCY": "Motorcycle",
    "APPORT": "Apportioned",
    "DISABL": "Disabled",
}

# County names printed on registration stickers. Matched only on the small text.
TEXAS_COUNTIES = {
    "STARR": "Starr County",
    "HIDALGO": "Hidalgo County",
    "CAMERON": "Cameron County",
    "WEBB": "Webb County",
    "ZAPATA": "Zapata County",
    "BEXAR": "Bexar County",
    "HARRIS": "Harris County",
    "DALLAS": "Dallas County",
    "TARRANT": "Tarrant County",
    "TRAVIS": "Travis County",
    "ELPASO": "El Paso County",
    "NUECES": "Nueces County",
    "COLLIN": "Collin County",
    "DENTON": "Denton County",
    "FORTBEND": "Fort Bend County",
    "WILLIAMSON": "Williamson County",
    "MONTGOMERY": "Montgomery County",
    "GALVESTON": "Galveston County",
    "BRAZORIA": "Brazoria County",
    "JEFFERSON": "Jefferson County",
    "LUBBOCK": "Lubbock County",
    "MIDLAND": "Midland County",
    "ECTOR": "Ector County",
    "SMITH": "Smith County",
    "BELL": "Bell County",
    "MCLENNAN": "McLennan County",
}

# Visual character confusion groups common in license-plate OCR under mud or motion blur.
OCR_CONFUSION_GROUPS = [
    set("0OQDC"),
    set("1ILT74/"),
    set("2Z"),
    set("5S"),
    set("6GB80"),
    set("8BR"),
    set("MNHKWR"),
]

DEFAULT_PASS_WEIGHTS = {
    "full_raw": 1.0,
    "band_raw": 1.0,
    "band_soft_clahe": 1.0,
    "sub_dark_ink": 1.0,
    "sticker_raw": 1.0,
}

_OCR_LOCK = threading.Lock()
_OCR_ENGINE = None
_OCR_LOOP = None
_CACHED_DETECTOR = None
_FEEDBACK_LOCK = threading.Lock()


def parse_args():
    """Read command-line options for plate reading and human-in-the-loop feedback."""
    parser = argparse.ArgumentParser(
        description="Enhance a license plate image, compute confidence, and learn from human feedback."
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="image paths inside this project. Default: photos in datasets/plates and datasets/",
    )
    parser.add_argument(
        "--confirm",
        "-y",
        action="store_true",
        help="confirm that the detected plate reading is accurate so the algorithm learns from it",
    )
    parser.add_argument(
        "--reject",
        "-n",
        action="store_true",
        help="mark the detected plate number as incorrect so the algorithm suppresses that token",
    )
    parser.add_argument(
        "--teach",
        metavar="PLATE_NUMBER",
        help="teach the algorithm the true plate number (e.g. --teach 291-764M)",
    )
    return parser.parse_args()


def resolve_inside_project(path):
    """Resolve an image path and refuse anything outside this project folder."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    resolved = resolved.resolve()
    if ROOT != resolved and ROOT not in resolved.parents:
        raise SystemExit(f"Refusing to read a path outside this folder: {resolved}")
    return resolved


def load_feedback():
    """Load the local human-feedback memory file from datasets/plate_feedback.json."""
    default = {
        "verified_plates": {},
        "rejected_tokens": {},
        "pass_weights": dict(DEFAULT_PASS_WEIGHTS),
    }
    with _FEEDBACK_LOCK:
        if not FEEDBACK_PATH.is_file():
            return default
        try:
            data = json.loads(FEEDBACK_PATH.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return default
            data.setdefault("verified_plates", {})
            data.setdefault("rejected_tokens", {})
            weights = dict(DEFAULT_PASS_WEIGHTS)
            weights.update(data.get("pass_weights") or {})
            data["pass_weights"] = weights
            return data
        except Exception:
            return default


def save_feedback(feedback):
    """Save updated human-feedback learning data inside datasets/plate_feedback.json."""
    with _FEEDBACK_LOCK:
        FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        FEEDBACK_PATH.write_text(json.dumps(feedback, indent=2), encoding="utf-8")


def _char_similarity(char_a, char_b):
    """Return 1.0 for identical characters, 0.85 for known OCR look-alikes, else 0.0."""
    if char_a == char_b:
        return 1.0
    for group in OCR_CONFUSION_GROUPS:
        if char_a in group and char_b in group:
            return 0.85
    return 0.0


def _collapse_ocr_splits(token):
    """Normalize common multi-stroke OCR splits (such as trailing '11'/'I1' for 'M')."""
    cleaned = re.sub(r"[^A-Z0-9]", "", token.upper())
    cleaned = re.sub(r"(11|I1|1I|II|MI)$", "M", cleaned)
    if len(cleaned) == 8 and re.match(r"^[A-Z17]\d{3}\d{3}[A-Z0-9]$", cleaned):
        cleaned = cleaned[1:]
    if len(cleaned) == 8 and re.match(r"^\d{3}[01479IS]\d{3}[A-Z0-9]$", cleaned):
        cleaned = cleaned[:3] + cleaned[4:]
    return cleaned


def ocr_similarity(candidate, target):
    """Compute OCR visual similarity in [0.0, 1.0] between two plate strings."""
    cand = _collapse_ocr_splits(candidate)
    targ = _collapse_ocr_splits(target)
    if not cand or not targ:
        return 0.0
    if cand == targ:
        return 1.0
    len_c, len_t = len(cand), len(targ)
    dp = [[0.0] * (len_t + 1) for _ in range(len_c + 1)]
    for i in range(1, len_c + 1):
        dp[i][0] = float(i)
    for j in range(1, len_t + 1):
        dp[0][j] = float(j)
    for i in range(1, len_c + 1):
        for j in range(1, len_t + 1):
            sub_cost = 1.0 - _char_similarity(cand[i - 1], targ[j - 1])
            dp[i][j] = min(
                dp[i - 1][j] + 1.0,
                dp[i][j - 1] + 1.0,
                dp[i - 1][j - 1] + sub_cost,
            )
    return max(0.0, 1.0 - (dp[len_c][len_t] / max(len_c, len_t)))


def discover_images():
    """Collect plate photos from datasets/plates and the top of datasets/."""
    found = []
    for folder in PLATE_DIRS:
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXT:
                found.append(path.resolve())
    unique = []
    seen = set()
    for path in found:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def read_image(path):
    """Load an image, including paths that contain spaces."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read image: {path}")
    return image


def enhance_plate_image(image):
    """Enhance a license plate image to recover readability under mud, glare, or low light.

    Suppresses warm brown/tan mud stains while preserving dark character ink,
    normalizes illumination with LAB percentile stretching and CLAHE, smooths
    grain with edge-preserving bilateral filtering, and sharpens text strokes.
    """
    b_chan, g_chan, r_chan = cv2.split(image.astype(np.float32))
    warmth = np.clip(r_chan - b_chan, 0.0, 255.0)
    luma = 0.114 * b_chan + 0.587 * g_chan + 0.299 * r_chan
    mud_mask = np.clip((warmth - 12.0) / 35.0, 0.0, 1.0) * np.clip(
        (luma - 40.0) / 60.0, 0.0, 1.0
    )
    lift = mud_mask * 90.0
    cleaned = cv2.merge(
        (
            np.clip(b_chan + lift * 1.25, 0, 255),
            np.clip(g_chan + lift * 1.05, 0, 255),
            np.clip(r_chan + lift * 0.85, 0, 255),
        )
    ).astype(np.uint8)

    lab = cv2.cvtColor(cleaned, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_lab = cv2.split(lab)
    p2, p98 = np.percentile(l_chan, (2, 98))
    if p98 - p2 > 10:
        l_stretched = np.clip(
            (l_chan.astype(np.float32) - p2) * 255.0 / (p98 - p2), 0, 255
        ).astype(np.uint8)
    else:
        l_stretched = l_chan
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l_eq = cv2.addWeighted(l_stretched, 0.60, clahe.apply(l_stretched), 0.40, 0)
    l_gamma = np.clip(((l_eq / 255.0) ** 1.35) * 255.0, 0, 255).astype(np.uint8)
    a_neutral = np.clip((a_chan.astype(np.float32) - 128.0) * 0.5 + 128.0, 0, 255).astype(np.uint8)
    b_neutral = np.clip((b_lab.astype(np.float32) - 128.0) * 0.5 + 128.0, 0, 255).astype(np.uint8)
    balanced = cv2.cvtColor(cv2.merge((l_gamma, a_neutral, b_neutral)), cv2.COLOR_LAB2BGR)
    denoised = cv2.bilateralFilter(balanced, 5, 30, 30)
    blurred = cv2.GaussianBlur(denoised, (0, 0), 1.3)
    return cv2.addWeighted(denoised, 1.45, blurred, -0.45, 0)


def _enhance_soft(crop):
    """Balance illumination and sharpen character strokes on a resized plate crop."""
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    p2, p98 = np.percentile(l_chan, (2, 98))
    if p98 - p2 > 10:
        l_chan = np.clip((l_chan.astype(np.float32) - p2) * 255.0 / (p98 - p2), 0, 255).astype(
            np.uint8
        )
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_eq = cv2.addWeighted(l_chan, 0.65, clahe.apply(l_chan), 0.35, 0)
    merged = cv2.cvtColor(cv2.merge((l_eq, a_chan, b_chan)), cv2.COLOR_LAB2BGR)
    denoised = cv2.bilateralFilter(merged, 5, 25, 25)
    blurred = cv2.GaussianBlur(denoised, (0, 0), 1.2)
    return cv2.addWeighted(denoised, 1.40, blurred, -0.40, 0)


def _enhance_dark_ink(crop):
    """Darken midtones so mud-covered or faded small print (like TRAILER) is legible."""
    return np.clip(((crop / 255.0) ** 1.8) * 255.0, 0, 255).astype(np.uint8)


def _recognize_bgr(bgr_image):
    """Run the cached local Windows OCR engine on a BGR image without re-initializing WinRT."""
    global _OCR_ENGINE, _OCR_LOOP
    from winrt.windows.globalization import Language
    from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
    from winrt.windows.media.ocr import OcrEngine
    from winrt.windows.storage.streams import DataWriter

    with _OCR_LOCK:
        if _OCR_ENGINE is None:
            lang = Language("en")
            if not OcrEngine.is_language_supported(lang):
                raise SystemExit(
                    "Windows English OCR is not installed. From an administrator PowerShell, run:\n"
                    "Add-WindowsCapability -Online -Name Language.OCR~~~en-US~0.0.1.0"
                )
            _OCR_ENGINE = OcrEngine.try_create_from_language(lang)
            _OCR_LOOP = asyncio.new_event_loop()

        bgra = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2BGRA)
        writer = DataWriter()
        writer.write_bytes(bgra.tobytes())
        bitmap = SoftwareBitmap.create_copy_from_buffer(
            writer.detach_buffer(),
            BitmapPixelFormat.BGRA8,
            bgr_image.shape[1],
            bgr_image.shape[0],
        )
        result = _OCR_LOOP.run_until_complete(_OCR_ENGINE.recognize_async(bitmap))
        words = []
        for line in result.lines:
            for word in line.words:
                rect = word.bounding_rect
                words.append((word.text, rect.x, rect.y, rect.width, rect.height))
        return words


def ocr_region(
    image, x0, y0, x1, y1, target_width, priority=1.0, enhance=None, pad=0, pass_name="raw"
):
    """Read words in one crop (with optional enhancement) and map boxes to the full image."""
    height, width = image.shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(width, int(x1)), min(height, int(y1))
    if x1 - x0 < 30 or y1 - y0 < 20:
        return []
    crop = image[y0:y1, x0:x1]
    scale = target_width / crop.shape[1]
    if scale <= 0:
        return []
    resized = cv2.resize(
        crop,
        (target_width, max(1, int(crop.shape[0] * scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    if enhance is not None:
        resized = enhance(resized)
    if pad > 0:
        resized = cv2.copyMakeBorder(
            resized, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
    words = _recognize_bgr(resized)
    items = []
    for raw_text, rx, ry, rw, rh in words:
        text = " ".join(str(raw_text).upper().split())
        if not text or rw <= 0:
            continue
        adj_x = max(0.0, float(rx) - pad)
        adj_y = max(0.0, float(ry) - pad)
        items.append(
            {
                "text": text,
                "x": x0 + adj_x / scale,
                "y": y0 + adj_y / scale,
                "w": float(rw) / scale,
                "h": float(rh) / scale,
                "cy": y0 + (adj_y + float(rh) / 2.0) / scale,
                "priority": float(priority),
                "pass_name": pass_name,
            }
        )
    return items


def dedupe(items):
    """Merge repeated words from overlapping crops while tracking multi-pass OCR agreement."""
    kept = []
    for item in sorted(
        items, key=lambda entry: (entry.get("priority", 1.0), entry["h"]), reverse=True
    ):
        token = compact(item["text"])
        if not token:
            continue
        pass_name = item.get("pass_name", "raw")
        duplicate = False
        for other in kept:
            same_text = token == compact(other["text"])
            same_row = abs(item["cy"] - other["cy"]) <= max(item["h"], other["h"])
            if same_text and same_row:
                other["passes"].add(pass_name)
                other["agreement_count"] = len(other["passes"])
                duplicate = True
                break
        if not duplicate:
            copied = dict(item)
            copied["passes"] = {pass_name}
            copied["agreement_count"] = 1
            kept.append(copied)
    return kept


def read_text(image, feedback=None):
    """OCR the whole plate plus enhanced bands, weighted by learned pass performance."""
    if feedback is None:
        feedback = load_feedback()
    weights = feedback.get("pass_weights") or DEFAULT_PASS_WEIGHTS
    height, width = image.shape[:2]
    band_width = 1000 if width >= 700 else 500
    passes = [
        # Full plate overview (captures state name, plate number, and bottom slogan)
        ("full_raw", 0, 0, width, height, 700, 2.0, None, 0),
        # Center number band (raw and contrast/denoise enhanced for dark/washed-out/muddy plates)
        ("band_raw", 0, height * 0.35, width, height * 0.78, band_width, 3.0, None, 0),
        (
            "band_soft_clahe",
            0,
            height * 0.35,
            width,
            height * 0.78,
            band_width,
            3.0,
            _enhance_soft,
            0,
        ),
        # Sub-header band with dark-ink gamma enhancement (recovers mud-covered 'TRAILER' etc.)
        (
            "sub_dark_ink",
            width * 0.32,
            height * 0.21,
            width * 0.64,
            height * 0.40,
            800,
            1.0,
            _enhance_dark_ink,
            30,
        ),
        # Registration sticker corner (captures county name, expiration year, and serial numbers)
        ("sticker_raw", width * 0.58, 0, width, height * 0.46, 900, 1.0, None, 0),
    ]
    items = []
    for pass_name, x0, y0, x1, y1, target_width, base_pri, enhance_fn, pad in passes:
        effective_pri = base_pri * float(weights.get(pass_name, 1.0))
        items.extend(
            ocr_region(
                image, x0, y0, x1, y1, target_width, effective_pri, enhance_fn, pad, pass_name
            )
        )
    return dedupe(items)


def detect_plate_regions(frame):
    """Locate rectangular license-plate candidate regions in a frame using OpenCV morphology.

    Works in ~2 ms by downscaling to 480px width, emphasizing dense horizontal
    dark-on-light character gradients (blackhat + Sobel-X), and filtering closed
    rectangular contours by aspect ratio.
    """
    height, width = frame.shape[:2]
    scale = min(1.0, 480.0 / max(width, 1))
    small = (
        cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else frame
    )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)
    grad_x = cv2.Sobel(blackhat, cv2.CV_16S, 1, 0, ksize=3)
    grad_x = cv2.convertScaleAbs(grad_x)
    blurred = cv2.GaussianBlur(grad_x, (5, 5), 0)
    closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, rect_kernel)
    _, thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw < 28 or ch < 10:
            continue
        aspect = cw / float(ch)
        if not 1.3 <= aspect <= 5.8:
            continue
        if cw > small.shape[1] * 0.88 and ch > small.shape[0] * 0.82:
            continue
        pad_x = int(cw * 0.06)
        pad_y = int(ch * 0.12)
        x0 = max(0, int((x - pad_x) / scale))
        y0 = max(0, int((y - pad_y) / scale))
        x1 = min(width, int((x + cw + pad_x) / scale))
        y1 = min(height, int((y + ch + pad_y) / scale))
        candidates.append((cw * ch, (x0, y0, x1, y1)))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [box for _, box in candidates[:3]]


def group_rows(items):
    """Group words that sit on the same horizontal line."""
    rows = []
    for item in sorted(items, key=lambda entry: entry["cy"]):
        placed = False
        for row in rows:
            if abs(item["cy"] - row["cy"]) <= 0.45 * min(item["h"], row["h"]):
                row["items"].append(item)
                heights = [entry["h"] for entry in row["items"]]
                row["h"] = float(np.median(heights))
                row["cy"] = float(np.mean([entry["cy"] for entry in row["items"]]))
                placed = True
                break
        if not placed:
            rows.append({"cy": item["cy"], "h": item["h"], "items": [item]})
    for row in rows:
        row["items"].sort(key=lambda entry: entry["x"])
    return rows


def join_row(row):
    """Join one line of words, keeping a space where the plate has a visual gap."""
    parts = []
    previous = None
    for item in row["items"]:
        if previous is not None:
            gap = item["x"] - (previous["x"] + previous["w"])
            if gap > 0.35 * max(previous["h"], item["h"]):
                parts.append(" ")
        parts.append(item["text"])
        previous = item
    return " ".join("".join(parts).split())


def compact(text):
    """Keep only letters and digits so '291-764M' and '291 764M' compare equal."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def find_state_and_slogan(rows, items):
    """Return (state, slogan) matched from the OCR text on the plate."""
    ordered_text = " ".join(join_row(row) for row in rows) + " " + " ".join(
        item["text"] for item in items
    )
    squashed = re.sub(r"[^A-Z0-9]", "", ordered_text)
    matched_state = None
    matched_slogan = None
    for raw_slogan, (state, display_slogan) in sorted(
        SLOGANS.items(), key=lambda pair: len(pair[0]), reverse=True
    ):
        if raw_slogan in ordered_text or re.sub(r"[^A-Z0-9]", "", raw_slogan) in squashed:
            matched_state = state
            matched_slogan = display_slogan
            break
    for name, state in sorted(STATE_NAMES.items(), key=lambda pair: len(pair[0]), reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", ordered_text):
            matched_state = state
            break
        if re.sub(r"[^A-Z]", "", name) in squashed and " " in name:
            matched_state = state
            break
    return matched_state, matched_slogan


def find_plate_type(items):
    """Return a plate classification such as Trailer when printed on the plate."""
    for item in items:
        alnum = compact(item["text"])
        if re.match(r"^TR[A4][I14L]L[E3]?R?$", alnum):
            return "Trailer"
        cleaned = re.sub(r"[^A-Z]", "", item["text"].upper())
        if cleaned in PLATE_TYPES:
            return PLATE_TYPES[cleaned]
        for prefix, label in PLATE_TYPE_PREFIXES.items():
            if cleaned.startswith(prefix) and len(cleaned) <= len(label) + 1:
                return label
    text = " ".join(item["text"] for item in items)
    for word, label in PLATE_TYPES.items():
        if re.search(rf"\b{word}\b", text):
            return label
    return None


def _is_non_number_text(text, letters, feedback=None):
    """Return True when a candidate is a state header, plate type, sticker date, or rejected token."""
    if feedback:
        rejected = feedback.get("rejected_tokens") or {}
        if letters in rejected:
            return True
        for v_entry in (feedback.get("verified_plates") or {}).values():
            if letters in (v_entry.get("sticker_numbers") or []):
                return True
    if letters in STATE_NAMES or letters in PLATE_TYPES or letters in TEXAS_COUNTIES:
        return True
    for name in STATE_NAMES:
        if len(name) >= 5 and name in letters:
            return True
    for kind in PLATE_TYPES:
        if len(kind) >= 5 and kind in letters:
            return True
    if re.match(r"^[TF7][EF][X4][A49][S5]", letters):
        return True
    if re.match(r"^[TY]R?[A4][I14L]L[E3]?R", letters):
        return True
    if re.search(r"\b\d{2}$", text.strip()) and len(letters) <= 7:
        return True
    return False


def _clean_shown_number(text):
    """Strip stray bracket/quote OCR artifacts and normalize separator silhouettes."""
    cleaned = text.strip("[](){}'\".,:;!?+-*/\\ ").replace(",", "-")
    token = compact(cleaned)
    if len(token) == 8 and "-" not in cleaned and re.match(r"^\d{3}[01479IS]\d{3}[A-Z]$", token):
        return f"{token[:3]}-{token[4:]}"
    if re.match(r"^\d{3}[^A-Z0-9]+\d{3}[A-Z]$", cleaned):
        return f"{token[:3]}-{token[3:]}"
    return cleaned


def _feedback_multiplier(letters, feedback):
    """Boost candidate score when a token matches or closely resembles a human-verified plate."""
    if not feedback:
        return 1.0
    verified = feedback.get("verified_plates") or {}
    if letters in verified:
        return 1.40
    for norm_key, entry in verified.items():
        if letters in (entry.get("aliases") or []):
            return 1.35
        if ocr_similarity(letters, norm_key) >= 0.84:
            return 1.28
    return 1.0


def number_score(row, feedback=None):
    """Score a text line as a plate number. Higher is a better candidate."""
    text = _clean_shown_number(join_row(row))
    letters = compact(text)
    if not 5 <= len(letters) <= 8:
        return -1
    if _is_non_number_text(text, letters, feedback):
        return -1
    if not any(character.isdigit() for character in letters):
        return -1
    score = row["h"] * max(item.get("priority", 1.0) for item in row["items"])
    if any(character.isalpha() for character in letters):
        score *= 1.25
    if 6 <= len(letters) <= 7:
        score *= 1.15
    if "-" in text:
        score *= 1.10
    score *= _feedback_multiplier(letters, feedback)
    return score


def item_number_score(item, feedback=None):
    """Score one word by itself, so a single '291-764M' token can win."""
    text = _clean_shown_number(item["text"])
    letters = compact(text)
    if not 5 <= len(letters) <= 8:
        return -1
    if _is_non_number_text(text, letters, feedback):
        return -1
    if not any(character.isdigit() for character in letters):
        return -1
    if not any(character.isalpha() for character in letters):
        return -1
    score = item["h"] * item.get("priority", 1.0) * 1.25
    if 6 <= len(letters) <= 7:
        score *= 1.15
    if "-" in text:
        score *= 1.10
    score *= _feedback_multiplier(letters, feedback)
    return score


def find_number(rows, items, feedback=None):
    """Pick the plate number from whole lines and from single words."""
    candidates = []
    for row in rows:
        score = number_score(row, feedback)
        if score > 0:
            shown = _clean_shown_number(join_row(row))
            candidates.append((score, shown, compact(shown), row["items"]))
    for item in items:
        score = item_number_score(item, feedback)
        if score > 0:
            shown = _clean_shown_number(item["text"])
            candidates.append((score, shown, compact(shown), [item]))
    if not candidates:
        return None, None, []
    _, shown, normalized, number_items = max(candidates, key=lambda entry: entry[0])
    return shown, normalized, number_items


def find_county(items, number_height, state):
    """Return a Texas county when its name is small sticker text, not the plate number."""
    if state != "Texas" or number_height <= 0:
        return None
    for item in items:
        if item["h"] > number_height * 0.7:
            continue
        token = compact(item["text"])
        if token in TEXAS_COUNTIES:
            return TEXAS_COUNTIES[token]
    return None


def find_sticker_details(rows, items, number_height, normalized_number):
    """Extract expiration date (such as 11/25) and registration sticker serial numbers."""
    expiration = None
    serials = []
    seen_serials = set()
    max_small_h = number_height * 0.65 if number_height > 0 else 9999.0

    for row in rows:
        row_items = [it for it in row["items"] if it["h"] <= max_small_h]
        for idx, item in enumerate(row_items):
            token = compact(item["text"])
            if len(token) == 2 and token.isdigit() and 20 <= int(token) <= 35:
                month = None
                if idx > 0:
                    prev_token = compact(row_items[idx - 1]["text"])
                    if prev_token.isdigit() and 1 <= int(prev_token) <= 12:
                        month = f"{int(prev_token):02d}"
                    elif prev_token in {"ALL", "LL", "II", "I1", "1I"} or prev_token.endswith("LL"):
                        month = "11"
                expiration = f"{month}/{token}" if month else f"20{token}"

    for item in sorted(items, key=lambda entry: (entry["y"], entry["x"])):
        if number_height > 0 and item["h"] > number_height * 0.45:
            continue
        token = compact(item["text"])
        if not 6 <= len(token) <= 10:
            continue
        if token == normalized_number or token in STATE_NAMES or token in PLATE_TYPES:
            continue
        if sum(ch.isdigit() for ch in token) < 5:
            continue
        if token not in seen_serials:
            seen_serials.add(token)
            serials.append(token)

    return expiration, serials


def _apply_learned_memory(reading, feedback):
    """Use human-verified plate memory to repair noisy OCR reads and backfill blurry contents."""
    norm = reading.get("normalized")
    if not norm or not feedback:
        return reading
    verified = feedback.get("verified_plates") or {}
    matched_key = None
    match_sim = 0.0
    if norm in verified:
        matched_key = norm
        match_sim = 1.0
    else:
        best_key, best_sim = None, 0.0
        for v_key, entry in verified.items():
            if norm in (entry.get("aliases") or []):
                best_key, best_sim = v_key, 0.96
                break
            sim = ocr_similarity(norm, v_key)
            if sim > best_sim:
                best_key, best_sim = v_key, sim
        if best_key and best_sim >= 0.84:
            matched_key = best_key
            match_sim = best_sim

    if not matched_key:
        return reading

    learned = verified[matched_key]
    reading["verified_match"] = matched_key
    reading["verified_similarity"] = round(match_sim, 3)
    reading["human_confirmations"] = int(learned.get("confirmations", 1))
    if norm != matched_key or reading["number"] != learned.get("number", reading["number"]):
        reading["raw_ocr_number"] = reading["number"]
        reading["number"] = learned.get("number") or reading["number"]
        reading["normalized"] = matched_key

    backfilled = []
    for field in ("state", "county", "plate_type", "slogan", "expiration"):
        if not reading.get(field) and learned.get(field):
            reading[field] = learned[field]
            backfilled.append(field)
    if reading.get("state") and not reading.get("country"):
        reading["country"] = "United States"
    if not reading.get("sticker_numbers") and learned.get("sticker_numbers"):
        reading["sticker_numbers"] = list(learned["sticker_numbers"])
        backfilled.append("sticker_numbers")
    if backfilled:
        reading["learned_backfill"] = backfilled
    return reading


def compute_confidence(reading, items, image=None):
    """Calculate a 0-100% accuracy confidence score and factor breakdown for the plate reading.

    Combines 5 signals:
      1. Multi-Pass OCR Agreement (up to 35 pts)
      2. Plate Format & Pattern Structure (up to 25 pts)
      3. Contextual Contents Corroboration (up to 20 pts)
      4. Visual Contrast & Edge Sharpness (up to 10 pts)
      5. Human-in-the-Loop Feedback Bonus (up to +15 pts)
    """
    norm = reading.get("normalized")
    if not norm:
        return 0.0, "None", {
            "agreement": 0.0,
            "format": 0.0,
            "context": 0.0,
            "clarity": 0.0,
            "human_feedback": 0.0,
        }

    # 1. Multi-Pass OCR Agreement (max 35.0)
    agreeing_passes = set()
    for item in reading.get("number_items") or []:
        agreeing_passes.update(item.get("passes") or {item.get("pass_name", "raw")})
    for item in items:
        if ocr_similarity(compact(item["text"]), norm) >= 0.85:
            agreeing_passes.update(item.get("passes") or {item.get("pass_name", "raw")})
    pass_count = len(agreeing_passes)
    if pass_count >= 3:
        agreement_pts = 35.0
    elif pass_count == 2:
        agreement_pts = 28.0
    else:
        agreement_pts = 18.0

    # 2. Alphanumeric Format & Pattern Validity (max 25.0)
    shown = reading.get("number") or norm
    format_pts = 0.0
    if 6 <= len(norm) <= 7:
        format_pts += 10.0
    elif 5 <= len(norm) <= 8:
        format_pts += 6.0
    if any(c.isdigit() for c in norm) and any(c.isalpha() for c in norm):
        format_pts += 6.0
    if re.match(r"^[A-Z0-9]{2,4}-[A-Z0-9]{3,4}$", shown):
        format_pts += 9.0
    elif re.match(r"^[A-Z0-9]{6,7}$", shown):
        format_pts += 7.0
    else:
        format_pts += 3.0
    if reading.get("raw_ocr_number"):
        # Slightly lower raw format points when repaired from a noisy OCR frame
        format_pts = max(12.0, format_pts - 4.0)

    # 3. Contextual Plate Contents Corroboration (max 20.0)
    backfilled = set(reading.get("learned_backfill") or [])
    context_pts = 0.0
    if reading.get("state") and "state" not in backfilled:
        context_pts += 7.0
    if reading.get("plate_type") and "plate_type" not in backfilled:
        context_pts += 4.0
    if reading.get("slogan") and "slogan" not in backfilled:
        context_pts += 4.0
    if (reading.get("county") and "county" not in backfilled) or (
        reading.get("expiration") and "expiration" not in backfilled
    ):
        context_pts += 5.0

    # 4. Visual Sharpness & Local Contrast (max 10.0)
    clarity_pts = 7.0
    num_items = reading.get("number_items") or []
    if image is not None and num_items:
        ih, iw = image.shape[:2]
        x0 = max(0, int(min(it["x"] for it in num_items)))
        y0 = max(0, int(min(it["y"] for it in num_items)))
        x1 = min(iw, int(max(it["x"] + it["w"] for it in num_items)))
        y1 = min(ih, int(max(it["y"] + it["h"] for it in num_items)))
        if x1 - x0 >= 20 and y1 - y0 >= 12:
            gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            contrast = min(5.0, float(np.std(gray)) / 11.0)
            gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
            sharpness = min(
                5.0,
                float(np.mean(cv2.convertScaleAbs(gx)) + np.mean(cv2.convertScaleAbs(gy))) / 18.0,
            )
            clarity_pts = round(contrast + sharpness, 1)

    # 5. Human-in-the-Loop Feedback Bonus (up to +15.0)
    confirmations = int(reading.get("human_confirmations", 0))
    feedback_pts = 0.0
    if confirmations > 0:
        sim = float(reading.get("verified_similarity", 1.0))
        base_boost = 8.0 if sim >= 0.99 else 6.0
        feedback_pts = min(15.0, round(base_boost + min(7.0, (confirmations - 1) * 1.5), 1))

    total = min(
        99.9,
        round(agreement_pts + format_pts + context_pts + clarity_pts + feedback_pts, 1),
    )
    if total >= 90.0:
        level = "Verified High" if confirmations > 0 else "High"
    elif total >= 75.0:
        level = "Good"
    elif total >= 55.0:
        level = "Moderate (Needs Human Check)"
    else:
        level = "Low (Uncertain)"

    details = {
        "agreement": round(agreement_pts, 1),
        "agreeing_passes": sorted(agreeing_passes),
        "format": round(format_pts, 1),
        "context": round(context_pts, 1),
        "clarity": round(clarity_pts, 1),
        "human_feedback": round(feedback_pts, 1),
        "confirmations": confirmations,
    }
    return total, level, details


def record_human_feedback(reading, items=None, is_correct=True, taught_number=None):
    """Update datasets/plate_feedback.json from human confirmation, rejection, or teaching.

    Returns the updated feedback dictionary and a short human-readable summary of what was learned.
    """
    feedback = load_feedback()
    items = items or reading.get("all_items") or []

    if not is_correct and not taught_number:
        bad_norm = reading.get("normalized")
        if not bad_norm:
            return feedback, "No plate number was present to reject."
        rejected = feedback.setdefault("rejected_tokens", {})
        rejected[bad_norm] = int(rejected.get(bad_norm, 0)) + 1
        verified = feedback.get("verified_plates") or {}
        if bad_norm in verified:
            del verified[bad_norm]
        save_feedback(feedback)
        return (
            feedback,
            f"Rejected '{reading.get('number') or bad_norm}' — added '{bad_norm}' to rejected tokens so the algorithm will ignore it.",
        )

    shown_number = _clean_shown_number(taught_number) if taught_number else reading.get("number")
    norm_number = compact(shown_number or "")
    if not norm_number:
        return feedback, "No valid plate number to confirm."

    # Remove from rejected list if previously rejected
    feedback.get("rejected_tokens", {}).pop(norm_number, None)

    verified = feedback.setdefault("verified_plates", {})
    entry = verified.get(
        norm_number,
        {
            "number": shown_number,
            "confirmations": 0,
            "aliases": [],
        },
    )
    entry["number"] = shown_number
    entry["confirmations"] = int(entry.get("confirmations", 0)) + 1

    for field in ("state", "county", "plate_type", "slogan", "expiration"):
        if reading.get(field):
            entry[field] = reading[field]
    if reading.get("sticker_numbers"):
        entry["sticker_numbers"] = list(reading["sticker_numbers"])

    # Learn OCR aliases from any noisy 5-8 char tokens seen in the image
    aliases = set(entry.get("aliases") or [])
    if reading.get("raw_ocr_number"):
        raw_norm = compact(reading["raw_ocr_number"])
        if raw_norm and raw_norm != norm_number:
            aliases.add(raw_norm)
    if reading.get("normalized") and reading["normalized"] != norm_number:
        aliases.add(reading["normalized"])
    for item in items:
        cand = compact(item.get("text", ""))
        if 5 <= len(cand) <= 9 and cand != norm_number:
            if not _is_non_number_text(item.get("text", ""), cand) and ocr_similarity(
                cand, norm_number
            ) >= 0.68:
                aliases.add(cand)
    entry["aliases"] = sorted(aliases)
    verified[norm_number] = entry

    # Reward enhancement passes that contributed to the winning plate reading
    weights = feedback.setdefault("pass_weights", dict(DEFAULT_PASS_WEIGHTS))
    winning_passes = set()
    for item in reading.get("number_items") or []:
        winning_passes.update(item.get("passes") or {item.get("pass_name", "raw")})
    for pass_name in winning_passes:
        if pass_name in weights:
            weights[pass_name] = round(min(1.50, float(weights[pass_name]) + 0.05), 3)

    save_feedback(feedback)
    return (
        feedback,
        f"Confirmed '{shown_number}' (confirmations={entry['confirmations']}, learned aliases={len(entry['aliases'])}).",
    )


def collect_plate_contents(reading, rows):
    """Build a clean, ordered list of all readable text contents printed on the plate."""
    contents = []
    seen = set()

    def _add(label):
        if not label:
            return
        key = compact(label)
        if key and key not in seen:
            seen.add(key)
            contents.append(label)

    if reading.get("state"):
        _add(reading["state"].upper())
    if reading.get("plate_type"):
        _add(reading["plate_type"].upper())
    if reading.get("number"):
        _add(reading["number"])
    if reading.get("slogan"):
        _add(reading["slogan"])
    if reading.get("county"):
        _add(reading["county"])
    if reading.get("expiration"):
        _add(f"EXP {reading['expiration']}")
    for serial in reading.get("sticker_numbers") or []:
        _add(serial)

    return contents


def parse_plate(items, image=None, feedback=None):
    """Turn OCR words into structured plate fields, learned corrections, and confidence score."""
    if feedback is None:
        feedback = load_feedback()
    rows = group_rows(items)
    shown, normalized, number_items = find_number(rows, items, feedback)
    number_height = max((item["h"] for item in number_items), default=0)
    state, slogan = find_state_and_slogan(rows, items)
    plate_type = find_plate_type(items)
    county = find_county(items, number_height, state)
    expiration, sticker_numbers = find_sticker_details(rows, items, number_height, normalized)

    valid_items = [
        item
        for item in items
        if len(compact(item["text"])) >= 2 and compact(item["text"]) not in {"ALL"}
    ]
    plate_box = None
    box_source = valid_items or number_items
    if box_source:
        x0 = int(min(item["x"] for item in box_source))
        y0 = int(min(item["y"] for item in box_source))
        x1 = int(max(item["x"] + item["w"] for item in box_source))
        y1 = int(max(item["y"] + item["h"] for item in box_source))
        plate_box = (x0, y0, x1, y1)

    reading = {
        "number": shown,
        "normalized": normalized,
        "state": state,
        "country": "United States" if state else None,
        "plate_type": plate_type,
        "slogan": slogan,
        "county": county,
        "expiration": expiration,
        "sticker_numbers": sticker_numbers,
        "number_items": number_items,
        "content_items": valid_items,
        "all_items": items,
        "plate_box": plate_box,
    }
    reading = _apply_learned_memory(reading, feedback)
    reading["contents"] = collect_plate_contents(reading, rows)
    confidence, conf_level, conf_details = compute_confidence(reading, items, image)
    reading["confidence"] = confidence
    reading["confidence_level"] = conf_level
    reading["confidence_details"] = conf_details
    return reading


def _shift_reading(reading, dx, dy):
    """Translate a crop reading's coordinates back onto the parent image."""
    for key in ("number_items", "content_items", "all_items"):
        for item in reading.get(key) or []:
            item["x"] += dx
            item["y"] += dy
            item["cy"] += dy
    if reading.get("plate_box"):
        px0, py0, px1, py1 = reading["plate_box"]
        reading["plate_box"] = (px0 + dx, py0 + dy, px1 + dx, py1 + dy)
    return reading


def read_plate_image(image, feedback=None):
    """Enhance and read one still photo. Returns the image and the parsed plate contents."""
    if feedback is None:
        feedback = load_feedback()
    items = read_text(image, feedback)
    reading = parse_plate(items, image=image, feedback=feedback)
    if reading["normalized"]:
        return image, reading

    # Fallback 1: run OCR on the full enhanced plate image
    enhanced = enhance_plate_image(image)
    enh_items = read_text(enhanced, feedback)
    enh_reading = parse_plate(enh_items, image=enhanced, feedback=feedback)
    if enh_reading["normalized"]:
        return image, enh_reading

    # Fallback 2: locate rectangular plate regions inside a wider scene
    for x0, y0, x1, y1 in detect_plate_regions(image):
        crop = image[y0:y1, x0:x1]
        crop_items = read_text(crop, feedback)
        crop_reading = parse_plate(crop_items, image=crop, feedback=feedback)
        if crop_reading["normalized"]:
            return image, _shift_reading(crop_reading, x0, y0)
    return image, reading


def read_plate_in_frame(image, vehicle_boxes=None):
    """Read a plate from a camera frame using fast plate localization, enhancement, and learned memory."""
    feedback = load_feedback()
    for x0, y0, x1, y1 in detect_plate_regions(image):
        crop = image[y0:y1, x0:x1]
        candidate = parse_plate(read_text(crop, feedback), image=crop, feedback=feedback)
        if candidate["normalized"]:
            return _shift_reading(candidate, x0, y0)

    reading = parse_plate(read_text(image, feedback), image=image, feedback=feedback)
    if reading["normalized"] or not vehicle_boxes:
        return reading

    height, width = image.shape[:2]
    for x0, y0, x1, y1 in vehicle_boxes[:2]:
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(width, int(x1)), min(height, int(y1))
        crop_y0 = y0 + int((y1 - y0) * 0.40)
        if x1 - x0 < 40 or y1 - crop_y0 < 30:
            continue
        crop = image[crop_y0:y1, x0:x1]
        candidate = parse_plate(read_text(crop, feedback), image=crop, feedback=feedback)
        if candidate["normalized"]:
            return _shift_reading(candidate, x0, crop_y0)
    return reading


def location_text(reading):
    """Format the issuing place printed on the plate, such as 'Starr County, Texas'."""
    if not reading or not reading.get("state"):
        return None
    if reading.get("county"):
        return f"{reading['county']}, {reading['state']}"
    return reading["state"]


def _get_vehicle_detector():
    """Load and fuse the local YOLOX model once for vehicle fallback crops."""
    global _CACHED_DETECTOR
    if _CACHED_DETECTOR is None:
        from camera import load_model

        _CACHED_DETECTOR = load_model(
            "yolox-s", ROOT / "weights" / "yolox_s.pth", 0.25, 0.45, "cpu"
        )
    return _CACHED_DETECTOR


def vehicle_crops(image):
    """Find car, truck, bus, and motorcycle crops with the local fused YOLOX model."""
    import torch
    from yolox.data.data_augment import ValTransform
    from yolox.data.datasets.coco_classes import COCO_CLASSES
    from yolox.utils.boxes import postprocess

    model, exp = _get_vehicle_detector()
    preproc = ValTransform(legacy=False)
    ratio = min(exp.test_size[0] / image.shape[0], exp.test_size[1] / image.shape[1])
    tensor, _ = preproc(image, None, exp.test_size)
    tensor = torch.from_numpy(tensor).unsqueeze(0).float()
    with torch.inference_mode():
        outputs = postprocess(
            model(tensor), exp.num_classes, exp.test_conf, exp.nmsthre, class_agnostic=True
        )
    if outputs[0] is None:
        return []
    detection = outputs[0].cpu()
    crops = []
    height, width = image.shape[:2]
    for row in detection:
        name = COCO_CLASSES[int(row[6])]
        if name not in {"car", "truck", "bus", "motorcycle"}:
            continue
        x0, y0, x1, y1 = [int(value / ratio) for value in row[:4]]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        if x1 - x0 < 40 or y1 - y0 < 40:
            continue
        crops.append(image[y0:y1, x0:x1])
    return crops


def annotate(image, reading):
    """Draw boxes around the plate contents, plate number, and a summary header with confidence."""
    canvas = image.copy()
    for item in reading.get("content_items") or []:
        ix0, iy0 = int(item["x"]), int(item["y"])
        ix1, iy1 = int(item["x"] + item["w"]), int(item["y"] + item["h"])
        cv2.rectangle(canvas, (ix0, iy0), (ix1, iy1), (220, 180, 40), 2)

    if reading["number_items"]:
        x0 = int(min(item["x"] for item in reading["number_items"]))
        y0 = int(min(item["y"] for item in reading["number_items"]))
        x1 = int(max(item["x"] + item["w"] for item in reading["number_items"]))
        y1 = int(max(item["y"] + item["h"] for item in reading["number_items"]))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 200, 0), 4)

    label_parts = []
    if reading["normalized"]:
        num_str = reading["number"] or reading["normalized"]
        conf = reading.get("confidence", 0.0)
        label_parts.append(f"{num_str} ({conf:.1f}%)")
    if reading.get("plate_type"):
        label_parts.append(reading["plate_type"])
    if reading["county"]:
        label_parts.append(reading["county"])
    elif reading["state"]:
        label_parts.append(reading["state"])
    if reading.get("expiration"):
        label_parts.append(f"Exp {reading['expiration']}")
    if label_parts:
        label = " | ".join(label_parts)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.82
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
        x, y = 12, text_height + 16
        cv2.rectangle(
            canvas,
            (x - 6, y - text_height - 8),
            (x + text_width + 6, y + baseline + 6),
            (0, 0, 0),
            -1,
        )
        cv2.putText(canvas, label, (x, y), font, scale, (80, 220, 80), thickness)
    return canvas


def save_image(path, image, suffix_tag=""):
    """Write an output photo under YOLOX_outputs/plates, keeping everything in this folder."""
    out_dir = ROOT / "YOLOX_outputs" / "plates"
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() if path.suffix else ".jpg"
    filename = f"{path.stem}{suffix_tag}{ext}" if suffix_tag else path.name
    destination = out_dir / filename
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        encoded = cv2.imencode(".jpg", image)[1]
        destination = destination.with_suffix(".jpg")
    encoded.tofile(str(destination))
    return destination


def describe(path, reading):
    """Print the plate number, confidence score breakdown, and all recognized plate contents."""
    print(path.relative_to(ROOT))
    if reading["normalized"]:
        extra = ""
        if reading.get("raw_ocr_number"):
            extra = f" (auto-corrected from noisy OCR '{reading['raw_ocr_number']}' via learned memory)"
        print(f"  Plate number: {reading['number']}{extra}")
    else:
        print("  Plate number: not found")

    conf = reading.get("confidence", 0.0)
    level = reading.get("confidence_level", "None")
    details = reading.get("confidence_details") or {}
    if reading["normalized"]:
        passes_str = ", ".join(details.get("agreeing_passes") or [])
        print(f"  Confidence score: {conf:.1f}% ({level})")
        print(
            f"    - OCR pass agreement: {details.get('agreement', 0.0):.1f}/35.0 pts "
            f"({len(details.get('agreeing_passes') or [])} passes: {passes_str})"
        )
        print(f"    - Plate format validity: {details.get('format', 0.0):.1f}/25.0 pts")
        print(f"    - Contextual contents match: {details.get('context', 0.0):.1f}/20.0 pts")
        print(f"    - Image clarity & contrast: {details.get('clarity', 0.0):.1f}/10.0 pts")
        print(
            f"    - Human feedback boost: +{details.get('human_feedback', 0.0):.1f}/15.0 pts "
            f"(confirmations: {details.get('confirmations', 0)})"
        )

    if reading["state"]:
        location = reading["state"]
        if reading["county"]:
            location = f"{reading['county']}, {location}"
        print(f"  Issued in: {location}, {reading['country']}")
    else:
        print("  Issued in: not printed clearly enough to tell")
    if reading.get("plate_type"):
        print(f"  Plate type: {reading['plate_type']}")
    if reading.get("slogan"):
        print(f"  Slogan: {reading['slogan']}")
    if reading.get("expiration"):
        print(f"  Sticker expiration: {reading['expiration']}")
    if reading.get("sticker_numbers"):
        print(f"  Sticker numbers: {', '.join(reading['sticker_numbers'])}")
    if reading.get("contents"):
        print(f"  Recognized contents: {' | '.join(reading['contents'])}")


def main():
    """Enhance and read every requested photo, compute confidence, and apply any human feedback."""
    args = parse_args()
    if args.images:
        paths = [resolve_inside_project(path) for path in args.images]
    else:
        (ROOT / "datasets" / "plates").mkdir(parents=True, exist_ok=True)
        paths = discover_images()
    if not paths:
        raise SystemExit(
            "No plate photos found.\n"
            "Put images in datasets\\plates, then run .\\run_plate.bat again."
        )

    for path in paths:
        if not path.is_file():
            print(f"Missing image: {path}")
            continue
        image = read_image(path)
        canvas, reading = read_plate_image(image)
        if not reading["normalized"]:
            for crop in vehicle_crops(image):
                crop_canvas, crop_reading = read_plate_image(crop)
                if crop_reading["normalized"]:
                    canvas, reading = crop_canvas, crop_reading
                    break

        if args.confirm or args.reject or args.teach:
            if args.reject:
                updated_fb, msg = record_human_feedback(reading, is_correct=False)
                print(f"  [Human Feedback] {msg}")
                canvas, reading = read_plate_image(image, feedback=updated_fb)
            else:
                updated_fb, msg = record_human_feedback(
                    reading, is_correct=True, taught_number=args.teach
                )
                print(f"  [Human Feedback] {msg}")
                canvas, reading = read_plate_image(image, feedback=updated_fb)

        describe(path, reading)
        enhanced_saved = save_image(path, enhance_plate_image(canvas), suffix_tag="_enhanced")
        saved = save_image(path, annotate(canvas, reading))
        print(f"  Enhanced photo: {enhanced_saved.relative_to(ROOT)}")
        print(f"  Marked photo: {saved.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
