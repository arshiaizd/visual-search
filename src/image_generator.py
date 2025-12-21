import os
import shutil
import random
import json
from PIL import Image, ImageDraw


class ObjectGenerator:
    def __init__(self, patch_size=28, grid_size=12, min_manhattan_dist=2):
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.min_manhattan_dist = min_manhattan_dist

    def draw_shape_in_patch(self, shape_name):
        img = Image.new("RGB", (self.patch_size, self.patch_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        side = int(self.patch_size * 0.75)
        half = side // 2
        cx = cy = self.patch_size // 2

        if "red" in shape_name:
            color = (255, 0, 0)
        elif "green" in shape_name:
            color = (0, 255, 0)
        else:
            raise ValueError(f"Unknown color in shape_name={shape_name!r}")

        box = [cx - half, cy - half, cx + half, cy + half]

        if "square" in shape_name:
            draw.rectangle(box, outline=color, width=2)
        elif "circle" in shape_name:
            draw.ellipse(box, outline=color, width=2)
        else:
            raise ValueError(f"Unknown shape in shape_name={shape_name!r}")

        return img

    def split_shape(self, shape_name):
        shape = "square" if "square" in shape_name else "circle"
        color = "green" if "green" in shape_name else "red"
        return shape, color

    def annotation_dict(self, filename, objects):
        """
        Option C: Preserve generation/list order, but make it explicit by
        storing a stable 'id' for each object (its index in the objects list
        at the time the annotation is written).
        """
        ann = {
            "image": filename,
            "grid": self.grid_size,
            "patch": self.patch_size,
            "objects": []
        }

        for obj_id, (shape_name, r, c) in enumerate(objects):
            shape, color = self.split_shape(shape_name)
            x1 = c * self.patch_size
            y1 = r * self.patch_size
            x2 = x1 + self.patch_size
            y2 = y1 + self.patch_size
            ann["objects"].append({
                "id": obj_id,  # <- explicit order/index
                "shape": shape,
                "color": color,
                "r": r,
                "c": c,
                "bbox": [x1, y1, x2, y2]
            })

        return ann

    def place_fixed(self, shape_name, count, used, placed):
        attempts = 0
        while count > 0 and attempts < 5000:
            attempts += 1
            r = random.randint(0, self.grid_size - 1)
            c = random.randint(0, self.grid_size - 1)

            if (r, c) in used:
                continue

            ok = True
            for ur, uc in used:
                if abs(r - ur) + abs(c - uc) < self.min_manhattan_dist:
                    ok = False
                    break
            if not ok:
                continue

            used.add((r, c))
            placed.append((shape_name, r, c))  # append order = generation order
            count -= 1

        if count > 0:
            raise RuntimeError(f"Could not place all {shape_name}")

    def draw_canvas(self, objects):
        full = self.patch_size * self.grid_size
        canvas = Image.new("RGB", (full, full), (255, 255, 255))

        for shape_name, r, c in objects:
            patch = self.draw_shape_in_patch(shape_name)
            canvas.paste(patch, (c * self.patch_size, r * self.patch_size))

        return canvas

    def draw_single_object_canvas(self, shape_name, r, c):
        full = self.patch_size * self.grid_size
        canvas = Image.new("RGB", (full, full), (255, 255, 255))
        patch = self.draw_shape_in_patch(shape_name)
        canvas.paste(patch, (c * self.patch_size, r * self.patch_size))
        return canvas


# -------------------- MAIN --------------------

if __name__ == "__main__":
    random.seed(42)

    GRID_SIZE = 12
    PATCH_SIZE = 28
    MANHATTAN_DIST = 2
    NUM_PAIRS = 200

    GREEN_SQUARE = "green_square"
    RED_CIRCLE = "red_circle"
    TARGET = "green_circle"

    # clean output
    if os.path.exists("data"):
        shutil.rmtree("data")

    os.makedirs("data/images", exist_ok=True)
    os.makedirs("data/isolated", exist_ok=True)

    gen = ObjectGenerator(
        patch_size=PATCH_SIZE,
        grid_size=GRID_SIZE,
        min_manhattan_dist=MANHATTAN_DIST
    )

    with open("data/annotations.jsonl", "w") as f:
        for pair_id in range(NUM_PAIRS):

            used = set()
            placed_base = []

            # 1) interior target location (reserved cell)
            while True:
                tr = random.randint(1, GRID_SIZE - 2)
                tc = random.randint(1, GRID_SIZE - 2)
                if (tr, tc) not in used:
                    break
            used.add((tr, tc))

            # 2) base objects (generation order is append order)
            gen.place_fixed(GREEN_SQUARE, 8, used, placed_base)
            gen.place_fixed(RED_CIRCLE, 8, used, placed_base)

            # with / without
            placed_with = placed_base + [(TARGET, tr, tc)]
            placed_without = placed_base + [
                (random.choice([GREEN_SQUARE, RED_CIRCLE]), tr, tc)
            ]

            for tag, placed in [("with", placed_with), ("without", placed_without)]:
                # main image
                img = gen.draw_canvas(placed)
                fname = f"data/images/pair_{pair_id:05d}_{tag}.png"
                img.save(fname)

                # annotation (includes explicit per-object id = list index)
                ann = gen.annotation_dict(fname, placed)
                ann.update({
                    "pair_id": pair_id,
                    "has_target": tag == "with",
                    "target": {"shape": "circle", "color": "green", "r": tr, "c": tc}
                })
                f.write(json.dumps(ann) + "\n")

                # ----- isolated objects -----
                folder = f"data/isolated/pair_{pair_id:05d}_{tag}"
                os.makedirs(folder, exist_ok=True)

                # IMPORTANT: keep isolated names aligned with annotation 'id'
                for obj_id, (shape_name, r, c) in enumerate(placed):
                    iso = gen.draw_single_object_canvas(shape_name, r, c)
                    iso_name = f"obj_{obj_id:02d}_{shape_name}_r{r}_c{c}.png"
                    iso.save(os.path.join(folder, iso_name))
