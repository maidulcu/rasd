#!/usr/bin/env python3
"""
Synthetic UAE Retail Dataset Generator.

Generates training images with UAE-market-relevant items composited
onto retail shelf backgrounds. Outputs YOLO-format labels.

Classes:
  0: dallah (Arabic coffee pot)
  1: perfume_bottle
  2: gold_box (jewelry box)
  3: electronics (phone/tablet)
  4: dates_box
  5: wallet
  6: watch
"""

import os, random, math, json
from pathlib import Path

import cv2
import numpy as np

random.seed(42)
np.random.seed(42)

ALL_CLASSES = [
    "dallah", "perfume_bottle", "gold_box",
    "electronics", "dates_box", "wallet", "watch",
]

CLASS_MAKERS = {
    "dallah": "make_dallah",
    "perfume_bottle": "make_perfume",
    "gold_box": "make_gold_box",
    "electronics": "make_electronics",
    "dates_box": "make_dates_box",
    "wallet": "make_wallet",
    "watch": "make_watch",
}

OUT_DIR = Path(__file__).parent / "data"
IMG_DIR = OUT_DIR / "images"
LAB_DIR = OUT_DIR / "labels"

N_TRAIN = 500
N_VAL = 100
IMG_SIZE = 640


# ── Item renderers ──────────────────────────────────────────────────────

def make_dallah():
    """Arabic coffee pot shape."""
    img = np.zeros((120, 80, 4), dtype=np.uint8)
    cv2.ellipse(img, (40, 75), (30, 35), 0, 0, 360, (50, 40, 30, 255), -1)
    cv2.rectangle(img, (30, 20), (50, 40), (50, 40, 30, 255), -1)
    pts = np.array([[50, 30], [75, 20], [75, 35]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (80, 70, 60, 255))
    cv2.ellipse(img, (20, 50), (10, 20), 0, -90, 90, (100, 90, 80, 255), 4)
    cv2.ellipse(img, (40, 18), (12, 6), 0, 0, 360, (180, 170, 150, 255), -1)
    return img


def make_perfume():
    """Perfume bottle with color variation."""
    w, h = 50, 110
    r, g, b = random.choices([30, 60, 100, 150, 200], k=3)
    img = np.zeros((h, w, 4), dtype=np.uint8)
    cv2.rectangle(img, (5, 20), (w-5, h-5), (r, g, b, 255), -1)
    cv2.rectangle(img, (5, 20), (w-5, h-5), (255, 255, 255, 80), 2)
    cv2.rectangle(img, (15, 5), (w-15, 22), (r, g, b, 255), -1)
    cv2.rectangle(img, (12, 0), (w-12, 8), (220, 200, 180, 255), -1)
    label_h = random.randint(15, 30)
    lr, lg, lb = 255-r, 255-g, 255-b
    cv2.rectangle(img, (8, h//2-label_h//2), (w-8, h//2+label_h//2), (lr, lg, lb, 200), -1)
    return img


def make_gold_box():
    """Gold/jewelry box."""
    w, h = 80, 60
    img = np.zeros((h, w, 4), dtype=np.uint8)
    gold = random.choice([(180, 160, 40), (200, 180, 60), (160, 140, 30)])
    cv2.rectangle(img, (2, 2), (w-2, h-2), gold + (255,), -1)
    pts = np.array([[2, 2], [w//2, 8], [w-2, 2], [w//2, -4]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (min(gold[0]+40,255), min(gold[1]+40,255), min(gold[2]+20,255), 255))
    cv2.rectangle(img, (w//2-12, 0), (w//2+12, h), (255, 50, 50, 200), -1)
    cv2.rectangle(img, (0, h//2-12), (w, h//2+12), (255, 50, 50, 200), -1)
    return img


def make_electronics():
    """Phone/tablet."""
    w, h = 55, 100
    img = np.zeros((h, w, 4), dtype=np.uint8)
    colors = [(20, 20, 30), (10, 10, 10), (30, 30, 50)]
    r, g, b = random.choice(colors)
    cv2.rectangle(img, (2, 2), (w-2, h-2), (r, g, b, 255), -1)
    cv2.rectangle(img, (6, 10), (w-6, h-10), (40, 60, 120, 255), -1)
    for i in range(3):
        ix, iy = 12 + i*14, h//2
        cv2.circle(img, (ix, iy), 4, (80, 200, 80, 200), -1)
    cv2.circle(img, (w//2, 6), 3, (50, 50, 50, 255), -1)
    return img


def make_dates_box():
    """Box of dates."""
    w, h = 90, 70
    img = np.zeros((h, w, 4), dtype=np.uint8)
    brown = random.choice([(80, 50, 30), (100, 65, 40), (60, 35, 20)])
    cv2.rectangle(img, (2, 2), (w-2, h-2), brown + (255,), -1)
    for _ in range(random.randint(6, 12)):
        dx = random.randint(8, w-8)
        dy = random.randint(8, h-8)
        cv2.circle(img, (dx, dy), random.randint(5, 9), (60, 40, 25, 255), -1)
        cv2.circle(img, (dx+1, dy-1), random.randint(2, 4), (100, 70, 40, 180), -1)
    cv2.rectangle(img, (w//2-20, h-15), (w//2+20, h-3), (255, 220, 150, 200), -1)
    return img


def make_wallet():
    """Wallet/purse."""
    w, h = 90, 65
    img = np.zeros((h, w, 4), dtype=np.uint8)
    colors = [(60, 40, 30), (30, 30, 50), (80, 30, 30)]
    r, g, b = random.choice(colors)
    cv2.rectangle(img, (2, 5), (w-2, h-5), (r, g, b, 255), -1)
    cv2.rectangle(img, (4, 5), (w-4, h-5), (min(r+40,255), min(g+40,255), min(b+40,255), 150), 1)
    cv2.rectangle(img, (2, 5), (w-2, h//3), (min(r+20,255), min(g+20,255), min(b+20,255), 255), -1)
    cv2.circle(img, (w//2, h//2-2), 5, (200, 180, 100, 255), -1)
    cv2.circle(img, (w//2, h//2-2), 3, (150, 130, 60, 255), -1)
    return img


def make_watch():
    """Watch."""
    w, h = 80, 90
    img = np.zeros((h, w, 4), dtype=np.uint8)
    band_c = random.choice([(40, 30, 30), (30, 30, 40), (50, 40, 30)])
    cv2.rectangle(img, (w//2-12, 2), (w//2+12, h-2), band_c + (255,), -1)
    cx, cy = w//2, h//3
    face_r = 22
    cv2.circle(img, (cx, cy), face_r, (220, 220, 230, 255), -1)
    cv2.circle(img, (cx, cy), face_r, (180, 180, 190, 255), 2)
    angle = random.randint(0, 360)
    for a, l in [(angle, 12), (angle+90, 8)]:
        ex = int(cx + l * math.cos(math.radians(a)))
        ey = int(cy + l * math.sin(math.radians(a)))
        cv2.line(img, (cx, cy), (ex, ey), (50, 50, 60, 255), 2)
    cv2.circle(img, (cx+face_r+4, cy-8), 4, (180, 170, 150, 255), -1)
    return img


ALL_ITEM_MAKERS = {
    "dallah": make_dallah,
    "perfume_bottle": make_perfume,
    "gold_box": make_gold_box,
    "electronics": make_electronics,
    "dates_box": make_dates_box,
    "wallet": make_wallet,
    "watch": make_watch,
}

ITEM_MAKERS = list(ALL_ITEM_MAKERS.values())


# ── Background generator ────────────────────────────────────────────────

def make_background(w=640, h=640):
    """Retail shelf-like background."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 240
    wall_c = random.randint(200, 240)
    img[:, :] = wall_c

    shelf_y = random.randint(200, 450)
    shelf_h = random.randint(3, 6)
    img[shelf_y:shelf_y+shelf_h, :] = random.choice([(160, 140, 120), (180, 160, 140), (140, 130, 120)])
    img[shelf_y-3:shelf_y, :] = (200, 190, 180)
    img[shelf_y+shelf_h:shelf_y+shelf_h+3, :] = (200, 190, 180)

    img[shelf_y+80:, :] = random.choice([(180, 170, 160), (190, 180, 170)])

    for _ in range(random.randint(2, 6)):
        ix = random.randint(50, w-100)
        iy = shelf_y - random.randint(20, 60)
        iw, ih = random.randint(15, 40), random.randint(20, 50)
        c = random.randint(100, 200)
        cv2.rectangle(img, (ix, iy-ih), (ix+iw, iy), (c-20, c-10, c), -1)
        cv2.rectangle(img, (ix, iy-ih), (ix+iw, iy), (c-40, c-30, c-20), 1)

    cv2.ellipse(img, (w//2, 0), (w//3, 30), 0, 0, 180, (255, 255, 255, 30), -1)

    return img


# ── Compositing ─────────────────────────────────────────────────────────

def composite(item_img, bg, max_instances=4):
    """Composite item images onto background, returns image + YOLO labels."""
    h, w = bg.shape[:2]
    result = bg.copy()
    labels = []

    n_instances = random.randint(1, max_instances)
    placed = []

    for _ in range(n_instances):
        for attempt in range(20):
            scale = random.uniform(0.8, 2.5)
            item = cv2.resize(item_img, (0, 0), fx=scale, fy=scale,
                              interpolation=cv2.INTER_LANCZOS4)
            ih, iw = item.shape[:2]

            if ih > h-20 or iw > w-20:
                continue

            x = random.randint(5, w - iw - 5)
            y = random.randint(5, h - ih - 5)

            new_box = [x, y, x+iw, y+ih]
            overlap = False
            for pb in placed:
                if iou(new_box, pb) > 0.15:
                    overlap = True
                    break
            if overlap:
                continue

            alpha = item[:, :, 3] / 255.0
            for c in range(3):
                result[y:y+ih, x:x+iw, c] = (
                    alpha * item[:, :, c] +
                    (1 - alpha) * result[y:y+ih, x:x+iw, c]
                )

            cx = (x + iw / 2) / w
            cy = (y + ih / 2) / h
            nw = iw / w
            nh = ih / h
            placed.append(new_box)
            labels.append(f"{cx} {cy} {nw} {nh}")
            break

    return result, labels


def iou(b1, b2):
    x1, y1, x2, y2 = b1
    x3, y3, x4, y4 = b2
    xi1, yi1 = max(x1, x3), max(y1, y3)
    xi2, yi2 = min(x2, x4), min(y2, y4)
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    a1 = (x2-x1)*(y2-y1)
    a2 = (x4-x3)*(y4-y3)
    return inter / (a1+a2-inter) if (a1+a2-inter) > 0 else 0


# ── Generation ──────────────────────────────────────────────────────────

def generate_split(n_images, start_id, class_names=None):
    """Generate n_images for specific classes (or all if None)."""
    if class_names:
        makers = [ALL_ITEM_MAKERS[c] for c in class_names]
        class_ids = [ALL_CLASSES.index(c) for c in class_names]
    else:
        makers = ITEM_MAKERS
        class_ids = list(range(len(ALL_CLASSES)))

    for i in range(n_images):
        img_id = start_id + i
        bg = make_background(IMG_SIZE, IMG_SIZE)
        result_img = bg.copy()
        all_labels = []

        # Pick 1-3 random item types from selected classes
        n_items = random.randint(1, min(3, len(makers)))
        chosen = random.sample(range(len(makers)), n_items)

        for idx in chosen:
            item_img = makers[idx]()
            real_cls_id = class_ids[idx]
            composed, labels = composite(item_img, result_img, max_instances=2)
            result_img = composed
            for l in labels:
                all_labels.append(f"{real_cls_id} {l}")

        img_path = IMG_DIR / f"img_{img_id:06d}.jpg"
        cv2.imwrite(str(img_path), result_img)

        if all_labels:
            lab_path = LAB_DIR / f"img_{img_id:06d}.txt"
            lab_path.write_text("\n".join(all_labels))

        if (i+1) % 100 == 0:
            print(f"  Generated {i+1}/{n_images}")


def generate_class_split(class_names, n_images, start_id):
    """Generate images for specific classes only, remapping IDs to 0..N."""
    # Build a local maker list and class mapping
    makers = [ALL_ITEM_MAKERS[c] for c in class_names]

    for i in range(n_images):
        img_id = start_id + i
        bg = make_background(IMG_SIZE, IMG_SIZE)
        result_img = bg.copy()
        all_labels = []

        n_items = random.randint(1, min(3, len(makers)))
        chosen = random.sample(range(len(makers)), n_items)

        for local_idx in chosen:
            item_img = makers[local_idx]()
            composed, labels = composite(item_img, result_img, max_instances=2)
            result_img = composed
            for l in labels:
                # Use LOCAL class ID (0, 1, 2...) so model only knows these classes
                all_labels.append(f"{local_idx} {l}")

        img_path = IMG_DIR / f"img_{img_id:06d}.jpg"
        cv2.imwrite(str(img_path), result_img)

        if all_labels:
            lab_path = LAB_DIR / f"img_{img_id:06d}.txt"
            lab_path.write_text("\n".join(all_labels))

        if (i+1) % 100 == 0:
            print(f"  Generated {i+1}/{n_images}")


def write_dataset_yaml(classes=None):
    """Write dataset.yaml for YOLO training."""
    if classes:
        names = classes
    else:
        names = ALL_CLASSES

    yaml_path = OUT_DIR.parent / "config.yaml"
    yaml_path.write_text(f"""# UAE Retail Dataset
path: {OUT_DIR.resolve()}
train: images
val: images

nc: {len(names)}
names: {json.dumps(names)}
""")
    print(f"  Wrote {yaml_path}")


if __name__ == "__main__":
    print("Generating UAE retail synthetic dataset...")
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(LAB_DIR, exist_ok=True)

    print(f"  Training images: {N_TRAIN}")
    generate_split(N_TRAIN, 0)

    print(f"  Validation images: {N_VAL}")
    generate_split(N_VAL, N_TRAIN)

    write_dataset_yaml()

    print(f"\nDone! Generated {N_TRAIN + N_VAL} images with {len(ALL_CLASSES)} classes.")
    print(f"  Classes: {', '.join(ALL_CLASSES)}")
