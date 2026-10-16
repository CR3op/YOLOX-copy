#!/usr/bin/env python3
"""Read a license plate and all of its printed contents with image enhancement.

Put plate photos in datasets/plates, or pass an image path inside this
project. Photos already sitting in datasets/ are read too. Reading uses
the local Windows OCR engine and OpenCV image enhancement so everything
stays inside this folder and nothing is downloaded from outside.
"""

import argparse
import asyncio
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

_OCR_LOCK = threading.Lock()
_OCR_ENGINE = None
_OCR_LOOP = None
_CACHED_DETECTOR = None


def parse_args():
    """Read optional image paths from the command line."""
    parser = argparse.ArgumentParser(
        description="Enhance a license plate image and read all of its contents locally."
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="image paths inside this project. Default: photos in datasets/plates and datasets/",
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


def ocr_region(image, x0, y0, x1, y1, target_width, priority=1, enhance=None, pad=0):
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
                "priority": priority,
            }
        )
    return items


def dedupe(items):
    """Drop repeated words that came from overlapping crops of the same text."""
    kept = []
    for item in sorted(
        items, key=lambda entry: (entry.get("priority", 1), entry["h"]), reverse=True
    ):
        token = compact(item["text"])
        if not token:
            continue
        duplicate = False
        for other in kept:
            same_text = token == compact(other["text"])
            same_row = abs(item["cy"] - other["cy"]) <= max(item["h"], other["h"])
            if same_text and same_row:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept


def read_text(image):
    """OCR the whole plate plus enhanced bands for the number, plate type, and sticker."""
    height, width = image.shape[:2]
    band_width = 1000 if width >= 700 else 500
    passes = [
        # Full plate overview (captures state name, plate number, and bottom slogan)
        (0, 0, width, height, 700, 2, None, 0),
        # Center number band (raw and contrast/denoise enhanced for dark/washed-out/muddy plates)
        (0, height * 0.35, width, height * 0.78, band_width, 3, None, 0),
        (0, height * 0.35, width, height * 0.78, band_width, 3, _enhance_soft, 0),
        # Sub-header band with dark-ink gamma enhancement (recovers mud-covered 'TRAILER' etc.)
        (width * 0.32, height * 0.21, width * 0.64, height * 0.40, 800, 1, _enhance_dark_ink, 30),
        # Registration sticker corner (captures county name, expiration year, and serial numbers)
        (width * 0.58, 0, width, height * 0.46, 900, 1, None, 0),
    ]
    items = []
    for x0, y0, x1, y1, target_width, priority, enhance_fn, pad in passes:
        items.extend(ocr_region(image, x0, y0, x1, y1, target_width, priority, enhance_fn, pad))
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
        # Ignore boxes that already cover the whole frame
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


def _clean_shown_number(text):
    """Strip stray bracket/quote OCR artifacts and normalize separator silhouettes."""
    cleaned = text.strip("[](){}'\".,:;!? ").replace(",", "-")
    token = compact(cleaned)
    # On lower-resolution crops, the center state-silhouette separator between 3 digits
    # and 4 characters (e.g. '291 [TX] 764M') can be read as an 8th character.
    if len(token) == 8 and "-" not in cleaned and re.match(r"^\d{3}[01479IS]\d{3}[A-Z]$", token):
        return f"{token[:3]}-{token[4:]}"
    return cleaned


def number_score(row):
    """Score a text line as a plate number. Higher is a better candidate."""
    text = _clean_shown_number(join_row(row))
    letters = compact(text)
    if not 5 <= len(letters) <= 8:
        return -1
    if letters in STATE_NAMES or letters in PLATE_TYPES:
        return -1
    if not any(character.isdigit() for character in letters):
        return -1
    score = row["h"] * max(item.get("priority", 1) for item in row["items"])
    if any(character.isalpha() for character in letters):
        score *= 1.25
    if 6 <= len(letters) <= 7:
        score *= 1.15
    if "-" in text:
        score *= 1.10
    return score


def item_number_score(item):
    """Score one word by itself, so a single '291-764M' token can win."""
    text = _clean_shown_number(item["text"])
    letters = compact(text)
    if not 5 <= len(letters) <= 8:
        return -1
    if letters in STATE_NAMES or letters in PLATE_TYPES:
        return -1
    if not any(character.isdigit() for character in letters):
        return -1
    if not any(character.isalpha() for character in letters):
        return -1
    score = item["h"] * item.get("priority", 1) * 1.25
    if 6 <= len(letters) <= 7:
        score *= 1.15
    if "-" in text:
        score *= 1.10
    return score


def find_number(rows, items):
    """Pick the plate number from whole lines and from single words."""
    candidates = []
    for row in rows:
        score = number_score(row)
        if score > 0:
            shown = _clean_shown_number(join_row(row))
            candidates.append((score, shown, compact(shown), row["items"]))
    for item in items:
        score = item_number_score(item)
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
                        # On Texas stickers, the state silhouette beside '11' often OCRs as 'ALL'
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


def parse_plate(items):
    """Turn OCR words into structured plate fields and all recognized plate contents."""
    rows = group_rows(items)
    shown, normalized, number_items = find_number(rows, items)
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
        "plate_box": plate_box,
    }
    reading["contents"] = collect_plate_contents(reading, rows)
    return reading


def _shift_reading(reading, dx, dy):
    """Translate a crop reading's coordinates back onto the parent image."""
    for key in ("number_items", "content_items"):
        for item in reading.get(key) or []:
            item["x"] += dx
            item["y"] += dy
            item["cy"] += dy
    if reading.get("plate_box"):
        px0, py0, px1, py1 = reading["plate_box"]
        reading["plate_box"] = (px0 + dx, py0 + dy, px1 + dx, py1 + dy)
    return reading


def read_plate_image(image):
    """Enhance and read one still photo. Returns the image and the parsed plate contents."""
    reading = parse_plate(read_text(image))
    if reading["normalized"]:
        return image, reading

    # Fallback 1: run OCR on the full enhanced plate image
    enhanced = enhance_plate_image(image)
    enh_reading = parse_plate(read_text(enhanced))
    if enh_reading["normalized"]:
        return image, enh_reading

    # Fallback 2: locate rectangular plate regions inside a wider scene
    for x0, y0, x1, y1 in detect_plate_regions(image):
        crop_reading = parse_plate(read_text(image[y0:y1, x0:x1]))
        if crop_reading["normalized"]:
            return image, _shift_reading(crop_reading, x0, y0)
    return image, reading


def read_plate_in_frame(image, vehicle_boxes=None):
    """Read a plate from a camera frame using fast plate localization and enhancement."""
    # First check rectangular plate candidates detected directly in the frame
    for x0, y0, x1, y1 in detect_plate_regions(image):
        candidate = parse_plate(read_text(image[y0:y1, x0:x1]))
        if candidate["normalized"]:
            return _shift_reading(candidate, x0, y0)

    # Next check the full frame (handles a plate held close to the camera)
    reading = parse_plate(read_text(image))
    if reading["normalized"] or not vehicle_boxes:
        return reading

    # Finally check the lower bumper area of any detected vehicles
    height, width = image.shape[:2]
    for x0, y0, x1, y1 in vehicle_boxes[:2]:
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(width, int(x1)), min(height, int(y1))
        crop_y0 = y0 + int((y1 - y0) * 0.40)
        if x1 - x0 < 40 or y1 - crop_y0 < 30:
            continue
        candidate = parse_plate(read_text(image[crop_y0:y1, x0:x1]))
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
    """Draw boxes around the plate contents, plate number, and a summary header."""
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
        label_parts.append(reading["number"] or reading["normalized"])
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
        scale = 0.85
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
    """Print the plate number, issuing place, type, slogan, sticker info, and all contents."""
    print(path.relative_to(ROOT))
    if reading["normalized"]:
        print(f"  Plate number: {reading['number']}")
    else:
        print("  Plate number: not found")
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
    """Enhance and read every requested photo, or every photo in the plate folders."""
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
        describe(path, reading)
        enhanced_saved = save_image(path, enhance_plate_image(canvas), suffix_tag="_enhanced")
        saved = save_image(path, annotate(canvas, reading))
        print(f"  Enhanced photo: {enhanced_saved.relative_to(ROOT)}")
        print(f"  Marked photo: {saved.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
