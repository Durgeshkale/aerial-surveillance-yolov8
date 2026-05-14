"""
Synthetic Aerial Surveillance Dataset Generator
Generates labeled aerial-view images with objects:
  0: person
  1: tarpaulin_gray
  2: tarpaulin_green
  3: artificial_net_2d
  4: artificial_net_3d
  5: artificial_grass_mat
  6: artificial_hedge
  7: structure
"""

import os
import random
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import math

CLASSES = {
    0: "person",
    1: "tarpaulin_gray",
    2: "tarpaulin_green",
    3: "artificial_net_2d",
    4: "artificial_net_3d",
    5: "artificial_grass_mat",
    6: "artificial_hedge",
    7: "structure",
}

IMG_SIZE = 640

def add_terrain_background(img, draw):
    """Creates a realistic aerial terrain background."""
    # Base earth tone
    base_color = random.choice([
        (139, 119, 101),  # brown earth
        (101, 130, 80),   # green field
        (180, 170, 140),  # sandy
        (90, 110, 75),    # dark grass
    ])
    img_arr = np.array(img)
    
    # Gradient base
    for y in range(IMG_SIZE):
        for x in range(0, IMG_SIZE, 8):
            noise = random.randint(-12, 12)
            r = max(0, min(255, base_color[0] + noise))
            g = max(0, min(255, base_color[1] + noise))
            b = max(0, min(255, base_color[2] + noise))
            img_arr[y, x:x+8] = [r, g, b]
    
    result = Image.fromarray(img_arr.astype(np.uint8))
    result = result.filter(ImageFilter.GaussianBlur(radius=1.5))
    return result

def draw_person(draw, cx, cy, scale=1.0):
    """Draw aerial top-down view of a person."""
    s = int(8 * scale)
    color = random.choice([(50,50,180),(180,50,50),(50,150,50),(200,150,50)])
    # body ellipse
    draw.ellipse([cx-s, cy-int(s*1.8), cx+s, cy+int(s*1.8)], fill=color)
    # head
    draw.ellipse([cx-s//2, cy-int(s*2.2), cx+s//2, cy-int(s*1.6)], fill=(220,180,140))
    return s*2, int(s*4.4)

def draw_tarpaulin(draw, cx, cy, color_type="gray", scale=1.0):
    """Draw tarpaulin from aerial view with folds."""
    w = int(random.randint(60, 120) * scale)
    h = int(random.randint(40, 90) * scale)
    angle = random.uniform(0, 360)
    
    base_color = (160,160,165) if color_type == "gray" else (60,120,60)
    fold_color = (140,140,145) if color_type == "gray" else (45,100,45)
    
    # Main rectangle (rotated)
    corners = get_rotated_rect(cx, cy, w, h, angle)
    draw.polygon(corners, fill=base_color)
    
    # Fold lines
    for i in range(1, 4):
        y_fold = cy - h//2 + i*(h//4)
        pt1, pt2 = rotate_point(cx-w//2, y_fold, cx, cy, angle), rotate_point(cx+w//2, y_fold, cx, cy, angle)
        draw.line([pt1, pt2], fill=fold_color, width=2)
    
    return w, h

def draw_net_2d(draw, cx, cy, scale=1.0):
    """Draw flat camouflage net."""
    w = int(random.randint(80, 150) * scale)
    h = int(random.randint(60, 120) * scale)
    color = (90, 110, 70)
    draw.rectangle([cx-w//2, cy-h//2, cx+w//2, cy+h//2], fill=color)
    # Net pattern
    for xi in range(cx-w//2, cx+w//2, 12):
        draw.line([(xi, cy-h//2), (xi, cy+h//2)], fill=(70,90,55), width=1)
    for yi in range(cy-h//2, cy+h//2, 12):
        draw.line([(cx-w//2, yi), (cx+w//2, yi)], fill=(70,90,55), width=1)
    return w, h

def draw_net_3d(draw, cx, cy, scale=1.0):
    """Draw 3D raised camouflage net with shadow."""
    w = int(random.randint(70, 130) * scale)
    h = int(random.randint(50, 100) * scale)
    # Shadow
    draw.ellipse([cx-w//2+5, cy-h//4+5, cx+w//2+5, cy+h//4+5], fill=(50,50,40))
    # Net surface
    draw.ellipse([cx-w//2, cy-h//4, cx+w//2, cy+h//4], fill=(100,120,75))
    # Texture
    for i in range(10):
        x1 = random.randint(cx-w//2, cx+w//2)
        y1 = random.randint(cy-h//4, cy+h//4)
        draw.ellipse([x1-3,y1-3,x1+3,y1+3], fill=(80,100,60))
    return w, h//2

def draw_grass_mat(draw, cx, cy, scale=1.0):
    """Draw artificial grass mat."""
    w = int(random.randint(80, 140) * scale)
    h = int(random.randint(60, 110) * scale)
    draw.rectangle([cx-w//2, cy-h//2, cx+w//2, cy+h//2], fill=(50,140,50))
    # Grass texture lines
    for i in range(20):
        x1 = random.randint(cx-w//2, cx+w//2)
        y1 = random.randint(cy-h//2, cy+h//2)
        draw.line([(x1, y1), (x1+random.randint(-5,5), y1-random.randint(3,8))], fill=(40,120,40), width=1)
    return w, h

def draw_hedge(draw, cx, cy, scale=1.0):
    """Draw artificial hedge (elongated green)."""
    w = int(random.randint(100, 200) * scale)
    h = int(random.randint(15, 30) * scale)
    draw.rectangle([cx-w//2, cy-h//2, cx+w//2, cy+h//2], fill=(30,100,30))
    for i in range(8):
        x1 = cx - w//2 + i*(w//8)
        draw.ellipse([x1, cy-h//2-3, x1+w//8, cy+h//2+3], fill=(40,120,40))
    return w, h

def draw_structure(draw, cx, cy, scale=1.0):
    """Draw building/structure from aerial view."""
    w = int(random.randint(50, 120) * scale)
    h = int(random.randint(50, 120) * scale)
    roof_color = random.choice([(180,80,80),(150,150,150),(200,180,100),(100,100,180)])
    draw.rectangle([cx-w//2, cy-h//2, cx+w//2, cy+h//2], fill=roof_color)
    # Shadow
    draw.polygon([(cx+w//2,cy-h//2),(cx+w//2+8,cy-h//2+8),(cx+w//2+8,cy+h//2+8),(cx+w//2,cy+h//2)], fill=(80,80,80))
    draw.polygon([(cx-w//2,cy+h//2),(cx+w//2,cy+h//2),(cx+w//2+8,cy+h//2+8),(cx-w//2+8,cy+h//2+8)], fill=(80,80,80))
    # Roof lines
    draw.line([(cx-w//2,cy-h//2),(cx+w//2,cy+h//2)], fill=(100,100,100), width=2)
    draw.line([(cx+w//2,cy-h//2),(cx-w//2,cy+h//2)], fill=(100,100,100), width=2)
    return w, h

def get_rotated_rect(cx, cy, w, h, angle_deg):
    angle = math.radians(angle_deg)
    corners = [(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)]
    rotated = []
    for (x,y) in corners:
        rx = x*math.cos(angle) - y*math.sin(angle) + cx
        ry = x*math.sin(angle) + y*math.cos(angle) + cy
        rotated.append((rx,ry))
    return rotated

def rotate_point(x, y, cx, cy, angle_deg):
    angle = math.radians(angle_deg)
    x -= cx; y -= cy
    rx = x*math.cos(angle) - y*math.sin(angle) + cx
    ry = x*math.sin(angle) + y*math.cos(angle) + cy
    return (rx, ry)

DRAWERS = {
    0: lambda d,cx,cy,s: draw_person(d,cx,cy,s),
    1: lambda d,cx,cy,s: draw_tarpaulin(d,cx,cy,"gray",s),
    2: lambda d,cx,cy,s: draw_tarpaulin(d,cx,cy,"green",s),
    3: lambda d,cx,cy,s: draw_net_2d(d,cx,cy,s),
    4: lambda d,cx,cy,s: draw_net_3d(d,cx,cy,s),
    5: lambda d,cx,cy,s: draw_grass_mat(d,cx,cy,s),
    6: lambda d,cx,cy,s: draw_hedge(d,cx,cy,s),
    7: lambda d,cx,cy,s: draw_structure(d,cx,cy,s),
}

def generate_image(img_id, out_img_dir, out_lbl_dir):
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (120,110,90))
    draw = ImageDraw.Draw(img)
    img = add_terrain_background(img, draw)
    draw = ImageDraw.Draw(img)

    annotations = []
    num_objects = random.randint(3, 10)
    placed = []

    for _ in range(num_objects):
        cls_id = random.randint(0, 7)
        scale = random.uniform(0.6, 1.4)
        margin = 80
        cx = random.randint(margin, IMG_SIZE - margin)
        cy = random.randint(margin, IMG_SIZE - margin)

        w, h = DRAWERS[cls_id](draw, cx, cy, scale)

        # YOLO format: class cx cy w h (normalized)
        x_norm = cx / IMG_SIZE
        y_norm = cy / IMG_SIZE
        w_norm = min(w / IMG_SIZE, 0.95)
        h_norm = min(h / IMG_SIZE, 0.95)
        annotations.append(f"{cls_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

    # Post-process: slight blur + noise to mimic aerial imagery
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))

    img_path = os.path.join(out_img_dir, f"aerial_{img_id:04d}.jpg")
    lbl_path = os.path.join(out_lbl_dir, f"aerial_{img_id:04d}.txt")
    img.save(img_path, quality=90)
    with open(lbl_path, "w") as f:
        f.write("\n".join(annotations))

    return img_path

def generate_dataset(n_train=300, n_val=80, n_test=40, base_dir="dataset"):
    splits = {"train": n_train, "val": n_val, "test": n_test}
    for split, count in splits.items():
        img_dir = os.path.join(base_dir, split, "images")
        lbl_dir = os.path.join(base_dir, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        print(f"Generating {count} {split} images...")
        for i in range(count):
            generate_image(i, img_dir, lbl_dir)
        print(f"  ✓ {split}: {count} images saved to {img_dir}")

    # Write dataset YAML
    yaml_content = f"""path: {os.path.abspath(base_dir)}
train: train/images
val: val/images
test: test/images

nc: {len(CLASSES)}
names: {list(CLASSES.values())}
"""
    with open(os.path.join(base_dir, "dataset.yaml"), "w") as f:
        f.write(yaml_content)
    print(f"\n✓ dataset.yaml written.")
    print("Dataset generation complete!")

if __name__ == "__main__":
    generate_dataset(n_train=300, n_val=80, n_test=40, base_dir="dataset")
