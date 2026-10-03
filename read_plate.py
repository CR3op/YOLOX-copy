#!/usr/bin/env python3
"""Read a license plate number and the state printed on it.

Put plate photos in datasets/plates, or pass an image path inside this
project. Photos already sitting in datasets/ are read too. Reading uses
the Windows OCR engine, so the model files stay on this PC.
"""

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
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

# Slogans that identify a state when the state name itself is hard to read.
SLOGANS = {
    "THE LONE STAR STATE": "Texas",
    "LONE STAR STATE": "Texas",
    "THE SUNSHINE STATE": "Florida",
    "THE GOLDEN STATE": "California",
    "EMPIRE STATE": "New York",
    "THE GARDEN STATE": "New Jersey",
    "GRAND CANYON STATE": "Arizona",
    "THE ALOHA STATE": "Hawaii",
    "THE LAST FRONTIER": "Alaska",
    "THE NATURAL STATE": "Arkansas",
    "LAND OF ENCHANTMENT": "New Mexico",
    "CENTENNIAL STATE": "Colorado",
    "CONSTITUTION STATE": "Connecticut",
    "THE FIRST STATE": "Delaware",
    "PEACH STATE": "Georgia",
    "GEM STATE": "Idaho",
    "PRAIRIE STATE": "Illinois",
    "HOOSIER": "Indiana",
    "HAWKEYE STATE": "Iowa",
    "SUNFLOWER STATE": "Kansas",
    "BLUEGRASS STATE": "Kentucky",
    "PELICAN STATE": "Louisiana",
    "PINE TREE STATE": "Maine",
    "OLD LINE STATE": "Maryland",
    "BAY STATE": "Massachusetts",
    "GREAT LAKES STATE": "Michigan",
    "NORTH STAR STATE": "Minnesota",
    "10,000 LAKES": "Minnesota",
    "MAGNOLIA STATE": "Mississippi",
    "SHOW ME STATE": "Missouri",
    "TREASURE STATE": "Montana",
    "CORNHUSKER STATE": "Nebraska",
    "SILVER STATE": "Nevada",
    "GRANITE STATE": "New Hampshire",
    "TAR HEEL STATE": "North Carolina",
    "PEACE GARDEN STATE": "North Dakota",
    "BUCKEYE STATE": "Ohio",
    "SOONER STATE": "Oklahoma",
    "BEAVER STATE": "Oregon",
    "KEYSTONE STATE": "Pennsylvania",
    "OCEAN STATE": "Rhode Island",
    "PALMETTO STATE": "South Carolina",
    "MOUNT RUSHMORE STATE": "South Dakota",
    "VOLUNTEER STATE": "Tennessee",
    "BEEHIVE STATE": "Utah",
    "GREEN MOUNTAIN STATE": "Vermont",
    "OLD DOMINION": "Virginia",
    "EVERGREEN STATE": "Washington",
    "MOUNTAIN STATE": "West Virginia",
    "BADGER STATE": "Wisconsin",
    "EQUALITY STATE": "Wyoming",
    "COWBOY STATE": "Wyoming",
}

# Words that describe the kind of plate, such as trailer or truck, not the number.
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
}

# County names printed on registration stickers. Matched only on the small text.
# County names that appear as small sticker text. Matched only when much smaller than the plate number.
# County names that appear as small sticker text. Matched only when much smaller than the plate number.
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


def parse_args():
    """Read optional image paths from the command line."""
    parser = argparse.ArgumentParser(
        description="Read the plate number and issuing state from a photo in this folder."
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


def ocr_region(image, x0, y0, x1, y1, target_width, priority=1):
    """Read words in one crop and map their boxes back onto the full image.

    Windows OCR misses plate numbers when the characters are very large, so
    each crop is scaled to target_width first. priority lets the big number
    band outrank sticker text of a similar size.
    """
    import winocr

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
        interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA,
    )
    try:
        result = winocr.recognize_cv2_sync(resized, "en")
    except AssertionError:
        raise SystemExit(
            "Windows English OCR is not installed. From an administrator PowerShell, run:\n"
            "Add-WindowsCapability -Online -Name Language.OCR~~~en-US~0.0.1.0"
        )
    items = []
    for line in result.get("lines") or []:
        for word in line.get("words") or []:
            text = " ".join(str(word.get("text", "")).upper().split())
            rect = word.get("bounding_rect") or {}
            if not text or "width" not in rect:
                continue
            items.append(
                {
                    "text": text,
                    "x": x0 + float(rect["x"]) / scale,
                    "y": y0 + float(rect["y"]) / scale,
                    "w": float(rect["width"]) / scale,
                    "h": float(rect["height"]) / scale,
                    "cy": y0 + (float(rect["y"]) + float(rect["height"]) / 2) / scale,
                    "priority": priority,
                }
            )
    return items


def dedupe(items):
    """Drop repeated words that came from overlapping crops of the same text."""
    kept = []
    for item in sorted(items, key=lambda entry: entry["h"], reverse=True):
        token = compact(item["text"])
        duplicate = False
        for other in kept:
            same_text = token and token == compact(other["text"])
            same_row = abs(item["cy"] - other["cy"]) <= max(item["h"], other["h"])
            if same_text and same_row:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept


def read_text(image):
    """OCR the whole plate plus the number band and the registration sticker."""
    height, width = image.shape[:2]
    # Each region is (x0, y0, x1, y1, scaled width, priority).
    # The center band is highest priority because that is where the plate number sits.
    regions = [
        (0, 0, width, height, 700, 1),
        (0, height * 0.35, width, height * 0.78, 1000, 3),
        (0, height * 0.28, width, height * 0.72, 1000, 2),
        (width * 0.58, 0, width, height * 0.46, 900, 1),
    ]
    items = []
    for x0, y0, x1, y1, target_width, priority in regions:
        items.extend(ocr_region(image, x0, y0, x1, y1, target_width, priority))
    return dedupe(items)


def group_rows(items):
    """Group words that sit on the same horizontal line.

    The gap test uses the shorter word so a tall plate number is not merged
    with the slogan underneath it.
    """
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


def find_state(items):
    """Return the US state named or sloganed in the OCR text."""
    text = " ".join(item["text"] for item in items)
    squashed = re.sub(r"[^A-Z]", "", text)
    for slogan, state in sorted(SLOGANS.items(), key=lambda pair: len(pair[0]), reverse=True):
        if slogan in text or re.sub(r"[^A-Z]", "", slogan) in squashed:
            return state
    for name, state in sorted(STATE_NAMES.items(), key=lambda pair: len(pair[0]), reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", text):
            return state
        if re.sub(r"[^A-Z]", "", name) in squashed and " " in name:
            return state
    return None


def find_plate_type(items):
    """Return a plate kind such as Trailer when that word is printed on it."""
    text = " ".join(item["text"] for item in items)
    for word, label in PLATE_TYPES.items():
        if re.search(rf"\b{word}\b", text):
            return label
    return None


def number_score(row):
    """Score a text line as a plate number. Higher is a better candidate.

    A plate number is 5 to 8 letters and digits, contains a digit, and is not
    a state name. Taller text and the center number band score higher.
    """
    text = join_row(row)
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
    return score


def item_number_score(item):
    """Score one word by itself, so a single '291-764M' token can win."""
    letters = compact(item["text"])
    if not 5 <= len(letters) <= 8:
        return -1
    if letters in STATE_NAMES or letters in PLATE_TYPES:
        return -1
    if not any(character.isdigit() for character in letters):
        return -1
    if not any(character.isalpha() for character in letters):
        return -1
    return item["h"] * item.get("priority", 1) * 1.25


def find_number(rows, items):
    """Pick the plate number from whole lines and from single words.

    Returns the display text, the letters-and-digits form, and the word boxes.
    """
    candidates = []
    for row in rows:
        score = number_score(row)
        if score > 0:
            shown = join_row(row).replace(",", "-")
            candidates.append((score, shown, compact(shown), row["items"]))
    for item in items:
        score = item_number_score(item)
        if score > 0:
            shown = item["text"].replace(",", "-")
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


def parse_plate(items):
    """Turn OCR words into a plate number, issuing state, county, and plate type."""
    rows = group_rows(items)
    shown, normalized, number_items = find_number(rows, items)
    number_height = max((item["h"] for item in number_items), default=0)
    state = find_state(items)
    return {
        "number": shown,
        "normalized": normalized,
        "state": state,
        "country": "United States" if state else None,
        "plate_type": find_plate_type(items),
        "county": find_county(items, number_height, state),
        "number_items": number_items,
    }


def read_plate_image(image):
    """Read one still photo. Returns the image and the parsed plate."""
    return image, parse_plate(read_text(image))


def read_plate_in_frame(image, vehicle_boxes=None):
    """Read a plate from a camera frame, then from the lower part of any vehicle.

    The second pass matters when the plate is on a car instead of filling the
    picture. Number boxes are shifted back onto the full frame.
    """
    reading = parse_plate(read_text(image))
    if reading["normalized"] or not vehicle_boxes:
        return reading
    height, width = image.shape[:2]
    for x0, y0, x1, y1 in vehicle_boxes[:2]:
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(width, int(x1)), min(height, int(y1))
        crop_y0 = y0 + int((y1 - y0) * 0.40)
        if x1 - x0 < 40 or y1 - crop_y0 < 30:
            continue
        candidate = parse_plate(read_text(image[crop_y0:y1, x0:x1]))
        if not candidate["normalized"]:
            continue
        for item in candidate["number_items"]:
            item["x"] += x0
            item["y"] += crop_y0
            item["cy"] += crop_y0
        return candidate
    return reading


def location_text(reading):
    """Format the issuing place printed on the plate, such as 'Starr County, Texas'."""
    if not reading or not reading.get("state"):
        return None
    if reading.get("county"):
        return f"{reading['county']}, {reading['state']}"
    return reading["state"]


def vehicle_crops(image):
    """Find car, truck, bus, and motorcycle crops with the local YOLOX model.

    Used only when a still photo has no readable plate in the full frame.
    """
    import torch
    from camera import load_model
    from yolox.data.data_augment import ValTransform
    from yolox.data.datasets.coco_classes import COCO_CLASSES
    from yolox.utils.boxes import postprocess

    model, exp = load_model("yolox-s", ROOT / "weights" / "yolox_s.pth", 0.25, 0.45, "cpu")
    preproc = ValTransform(legacy=False)
    ratio = min(exp.test_size[0] / image.shape[0], exp.test_size[1] / image.shape[1])
    tensor, _ = preproc(image, None, exp.test_size)
    tensor = torch.from_numpy(tensor).unsqueeze(0).float()
    with torch.no_grad():
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
    """Draw a box around the plate number and a label with the number and place."""
    canvas = image.copy()
    if reading["number_items"]:
        x0 = int(min(item["x"] for item in reading["number_items"]))
        y0 = int(min(item["y"] for item in reading["number_items"]))
        x1 = int(max(item["x"] + item["w"] for item in reading["number_items"]))
        y1 = int(max(item["y"] + item["h"] for item in reading["number_items"]))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 180, 0), 3)
    label_parts = []
    if reading["normalized"]:
        label_parts.append(reading["number"] or reading["normalized"])
    if reading["county"]:
        label_parts.append(reading["county"])
    elif reading["state"]:
        label_parts.append(reading["state"])
    if label_parts:
        label = " | ".join(label_parts)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.9
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


def save_image(path, image):
    """Write the marked photo under YOLOX_outputs/plates, keeping the original name."""
    out_dir = ROOT / "YOLOX_outputs" / "plates"
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / path.name
    ok, encoded = cv2.imencode(path.suffix.lower() if path.suffix else ".jpg", image)
    if not ok:
        encoded = cv2.imencode(".jpg", image)[1]
        destination = destination.with_suffix(".jpg")
    encoded.tofile(str(destination))
    return destination


def describe(path, reading):
    """Print the plate number, issuing place, and plate type for one photo."""
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
    if reading["plate_type"]:
        print(f"  Plate type: {reading['plate_type']}")


def main():
    """Read every requested photo, or every photo in the plate folders."""
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
        saved = save_image(path, annotate(canvas, reading))
        print(f"  Marked photo: {saved.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
