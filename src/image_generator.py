import os
import shutil
import random
import json
from PIL import Image, ImageDraw


class ObjectGenerator:
    def __init__(self, patch_size=28, grid_size=12, min_manhattan_dist=2, shape_scale=1):
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.min_manhattan_dist = min_manhattan_dist
        self.shape_scale = shape_scale  # New parameter: how many patches per shape (scale factor)

    def draw_shape(self, shape_name: str) -> Image.Image:
        """Draw a shape that may span multiple patches."""
        # Canvas size scales with shape_scale
        canvas_size = self.patch_size * self.shape_scale
        img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Scale shape size proportionally
        side = int(self.patch_size * 0.75 * self.shape_scale)
        half = side // 2
        cx = cy = canvas_size // 2

        if "red" in shape_name:
            color = (255, 0, 0)
        elif "green" in shape_name:
            color = (0, 255, 0)
        else:
            raise ValueError(f"Unknown color in shape_name={shape_name!r}")

        box = [cx - half, cy - half, cx + half, cy + half]

        if "square" in shape_name:
            draw.rectangle(box, outline=color, width=5)
        elif "circle" in shape_name:
            draw.ellipse(box, outline=color, width=5)
        else:
            raise ValueError(f"Unknown shape in shape_name={shape_name!r}")

        return img

    def split_shape(self, shape_name: str):
        shape = "square" if "square" in shape_name else "circle"
        color = "green" if "green" in shape_name else "red"
        return shape, color

    def annotation_dict(self, filename: str, objects):
        """Preserve generation/list order and store explicit 'id' that matches isolated image index."""
        ann = {
            "image": filename,
            "grid": self.grid_size,
            "patch": self.patch_size,
            "shape_scale": self.shape_scale,  # New: record shape scale
            "objects": []
        }

        for obj_id, (shape_name, r, c) in enumerate(objects):
            shape, color = self.split_shape(shape_name)
            # Bounding box now spans multiple patches
            x1 = c * self.patch_size
            y1 = r * self.patch_size
            x2 = x1 + self.patch_size * self.shape_scale
            y2 = y1 + self.patch_size * self.shape_scale
            ann["objects"].append({
                "id": obj_id,
                "shape": shape,
                "color": color,
                "r": r,  # Top-left row
                "c": c,  # Top-left column
                "bbox": [x1, y1, x2, y2]
            })

        return ann

    def place_fixed(self, shape_name: str, count: int, used: set, placed: list):
        """Place shapes with multi-patch footprints."""
        attempts = 0
        while count > 0 and attempts < 5000:
            attempts += 1
            
            # Ensure shape fits within grid bounds
            max_r = self.grid_size - self.shape_scale
            max_c = self.grid_size - self.shape_scale
            if max_r < 0 or max_c < 0:
                raise ValueError(f"shape_scale {self.shape_scale} too large for grid_size {self.grid_size}")
            
            r = random.randint(0, max_r)
            c = random.randint(0, max_c)

            # Calculate all cells this shape occupies
            footprint = [(r + dr, c + dc) for dr in range(self.shape_scale) for dc in range(self.shape_scale)]

            # Check if any cell in footprint is already used
            if any(cell in used for cell in footprint):
                continue

            # Enforce min Manhattan distance between ANY cells of different shapes
            ok = True
            for (fr, fc) in footprint:
                for (ur, uc) in used:
                    dist = abs(fr - ur) + abs(fc - uc)
                    if dist < self.min_manhattan_dist:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue

            # All checks passed: mark footprint as used
            used.update(footprint)
            placed.append((shape_name, r, c))
            count -= 1

        if count > 0:
            raise RuntimeError(f"Could not place all {shape_name} (remaining={count})")

    def draw_canvas(self, objects) -> Image.Image:
        """Draw full canvas with multi-patch shapes."""
        full = self.patch_size * self.grid_size
        canvas = Image.new("RGB", (full, full), (255, 255, 255))

        for shape_name, r, c in objects:
            patch = self.draw_shape(shape_name)  # Renamed method
            # Paste at top-left corner of shape's footprint
            canvas.paste(patch, (c * self.patch_size, r * self.patch_size))

        return canvas

    def draw_single_object_canvas(self, shape_name: str, r: int, c: int) -> Image.Image:
        """Draw canvas with single multi-patch shape."""
        full = self.patch_size * self.grid_size
        canvas = Image.new("RGB", (full, full), (255, 255, 255))
        patch = self.draw_shape(shape_name)  # Renamed method
        canvas.paste(patch, (c * self.patch_size, r * self.patch_size))
        return canvas


# -------------------- MAIN --------------------

if __name__ == "__main__":
    random.seed(42)

    # Configuration
    GRID_SIZE = 24
    PATCH_SIZE = 28
    MANHATTAN_DIST = 3
    NUM_PAIRS = 200
    SHAPE_SCALE = 2  # New: each shape occupies 2x2 patches
    have_with = True

    GREEN_SQUARE = "green_square"
    RED_CIRCLE = "red_circle"
    TARGET = "green_circle"

    # Clean output
    directory_name = "data"
    if os.path.exists(directory_name):
        shutil.rmtree(directory_name)
    os.makedirs(directory_name + "/images", exist_ok=True)
    os.makedirs(directory_name + "/isolated", exist_ok=True)

    gen = ObjectGenerator(
        patch_size=PATCH_SIZE,
        grid_size=GRID_SIZE,
        min_manhattan_dist=MANHATTAN_DIST,
        shape_scale=SHAPE_SCALE  # New parameter
    )

    with open(directory_name + "/annotations.jsonl", "w") as f:
        for pair_id in range(NUM_PAIRS):
            used = set()
            placed_base = []

            # Reserve interior footprint for target location (kept empty in "without")
            # Ensure footprint fits and is at least 1 cell away from border (like original)
            min_coord = 1
            max_coord = GRID_SIZE - SHAPE_SCALE - 1  # Ensure footprint stays interior
            while True:
                tr = random.randint(min_coord, max_coord)
                tc = random.randint(min_coord, max_coord)
                target_footprint = [(tr + dr, tc + dc) for dr in range(SHAPE_SCALE) for dc in range(SHAPE_SCALE)]
                if not any(cell in used for cell in target_footprint):
                    used.update(target_footprint)
                    break

            # Place base objects (total 16 objects)
            gen.place_fixed(GREEN_SQUARE, 8, used, placed_base)
            gen.place_fixed(RED_CIRCLE, 8, used, placed_base)

            placed_without = placed_base
            placed_with = list(placed_base)

            # Find an interior red_circle to replace with target
            idx_to_replace = None
            for i, (shape_name, r, c) in enumerate(placed_with):
                if shape_name == RED_CIRCLE:
                    # Check if shape footprint is interior
                    is_interior = (1 <= r <= GRID_SIZE - SHAPE_SCALE - 1) and (1 <= c <= GRID_SIZE - SHAPE_SCALE - 1)
                    if is_interior:
                        idx_to_replace = i
                        break

            # Fallback: replace any red_circle if no interior found
            if idx_to_replace is None:
                for i, (shape_name, r, c) in enumerate(placed_with):
                    if shape_name == RED_CIRCLE:
                        idx_to_replace = i
                        break

            if idx_to_replace is None:
                raise RuntimeError("No red_circle found to replace")

            # Get location of shape to replace and update with target
            _, tr, tc = placed_with[idx_to_replace]
            placed_with[idx_to_replace] = (TARGET, tr, tc)

            if have_with:
                options = [("without", placed_without), ("with", placed_with)]
            else:
                options = [("without", placed_without)]

            for tag, placed in options:
                # Main image
                img = gen.draw_canvas(placed)

                if have_with:
                    image_id = pair_id * 2 + (1 if tag == "with" else 0)
                else:
                    image_id = pair_id
                fname = directory_name + f"/images/image_{image_id:03d}.png"
                img.save(fname)

                # Annotation
                ann = gen.annotation_dict(fname, placed)
                ann.update({
                    "image_id": image_id,
                    "pair_id": pair_id,
                    "has_target": (tag == "with"),
                    "target": {"shape": "circle", "color": "green", "r": tr, "c": tc}
                })
                f.write(json.dumps(ann) + "\n")

                # Isolated objects
                folder = directory_name + f"/isolated/image_{image_id:05d}"
                os.makedirs(folder, exist_ok=True)

                # Isolated object ids match annotation ids
                # for obj_id, (shape_name, r, c) in enumerate(placed):
                #     iso = gen.draw_single_object_canvas(shape_name, r, c)
                #     iso_name = f"obj_{obj_id:02d}_{shape_name}_r{r}_c{c}.png"
                #     iso.save(os.path.join(folder, iso_name))