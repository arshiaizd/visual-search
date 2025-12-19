import os
import shutil
import random
import json
import math
from PIL import Image, ImageDraw

class ObjectGenerator:
    def __init__(self, patch_size=28, grid_size=12, min_manhattan_dist=1,
                 allowed_shapes=None, max_counts=None, min_counts=None):
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.min_manhattan_dist = min_manhattan_dist

        all_shapes = [
            "red_square", "green_square", "blue_square", "yellow_square",
            "red_circle", "green_circle", "blue_circle", "yellow_circle",
            "red_triangle", "green_triangle", "blue_triangle", "yellow_triangle",
            "red_star", "green_star", "blue_star", "yellow_star"
        ]
        self.allowed_shapes = allowed_shapes if allowed_shapes is not None else all_shapes

        self.max_counts = max_counts if max_counts is not None else {s: None for s in self.allowed_shapes}
        for s in self.allowed_shapes:
            if s not in self.max_counts:
                self.max_counts[s] = None

        self.min_counts = min_counts if min_counts is not None else {s: 0 for s in self.allowed_shapes}
        for s in self.allowed_shapes:
            if s not in self.min_counts:
                self.min_counts[s] = 0


    def draw_shape_in_patch(self, shape_name):
        img = Image.new("RGB", (self.patch_size, self.patch_size), (255,255,255))
        draw = ImageDraw.Draw(img)
        side = int(self.patch_size * 0.75)
        half = side // 2
        cx = cy = self.patch_size // 2

        # ---- color ----
        if "red" in shape_name:
            color = (255, 0, 0)
        elif "green" in shape_name:
            color = (0, 255, 0)
        elif "blue" in shape_name:
            color = (0, 0, 255)
        elif "yellow" in shape_name:
            color = (255, 255, 0)
        else:
            color = (0, 0, 0)

        box = [cx-half, cy-half, cx+half, cy+half]

        # ---- shape ----
        if "square" in shape_name:
            draw.rectangle(box, outline=color, width=2)
        elif "circle" in shape_name:
            draw.ellipse(box, outline=color, width=2)
        elif "triangle" in shape_name:
            # Upright triangle
            p1 = (cx, cy - half)
            p2 = (cx - half, cy + half)
            p3 = (cx + half, cy + half)
            draw.polygon([p1, p2, p3], outline=color, width=2)
        elif "star" in shape_name:
            # 5-pointed star
            outer_r = half
            inner_r = int(half * 0.5)
            points = []
            for k in range(10):
                angle = math.radians(-90 + k * 36)
                r = outer_r if k % 2 == 0 else inner_r
                x = cx + int(r * math.cos(angle))
                y = cy + int(r * math.sin(angle))
                points.append((x, y))
            draw.polygon(points, outline=color, width=2)
        else:
            # fallback: circle
            raise NotImplementedError(f"The specified shape ({shape_name}) is not implemented")

        return img

    def split_shape(self, shape_name):
        if "square" in shape_name:
            shape = "square"
        elif "circle" in shape_name:
            shape = "circle"
        elif "triangle" in shape_name:
            shape = "triangle"
        elif "star" in shape_name:
            shape = "star"
        else:
            shape = "unknown"

        if "red" in shape_name:
            color = "red"
        elif "green" in shape_name:
            color = "green"
        elif "blue" in shape_name:
            color = "blue"
        elif "yellow" in shape_name:
            color = "yellow"
        else:
            color = "unknown"

        return shape, color

    def annotation_dict(self, filename, objects_list):
        ann = {"image": filename, "grid": self.grid_size, "patch": self.patch_size, "objects": []}
        for (shape_name, r, c) in objects_list:
            shape, color = self.split_shape(shape_name)
            x1 = c * self.patch_size; y1 = r * self.patch_size
            x2 = x1 + self.patch_size; y2 = y1 + self.patch_size
            ann["objects"].append({"shape": shape, "color": color, "r": r, "c": c, "bbox": [x1, y1, x2, y2]})
        return ann

    def generate(self, num_objects):
        used = []
        shape_counts = {s: 0 for s in self.allowed_shapes}
        chosen_objects = []
        attempts = 0

        # ---- STAGE 1: satisfy min_counts ----
        for shape_name in self.allowed_shapes:
            required = self.min_counts[shape_name]

            while shape_counts[shape_name] < required and attempts < 5000:
                attempts += 1
                i = random.randint(0, self.grid_size - 1)
                j = random.randint(0, self.grid_size - 1)
                if (i, j) in used:
                    continue

                # Manhattan dist check
                ok = True
                for (pi, pj) in used:
                    if abs(i - pi) + abs(j - pj) < self.min_manhattan_dist:
                        ok = False
                        break
                if not ok:
                    continue

                used.append((i, j))
                chosen_objects.append((shape_name, i, j))
                shape_counts[shape_name] += 1

        # ---- STAGE 2: fill remaining objects normally ----
        while len(chosen_objects) < num_objects and attempts < 5000:
            attempts += 1
            shape_name = random.choice(self.allowed_shapes)

            # respect max limits
            mx = self.max_counts[shape_name]
            if mx is not None and shape_counts[shape_name] >= mx:
                continue

            i = random.randint(0, self.grid_size - 1)
            j = random.randint(0, self.grid_size - 1)
            if (i, j) in used:
                continue

            ok = True
            for (pi, pj) in used:
                if abs(i - pi) + abs(j - pj) < self.min_manhattan_dist:
                    ok = False
                    break
            if not ok:
                continue

            used.append((i, j))
            chosen_objects.append((shape_name, i, j))
            shape_counts[shape_name] += 1

        # ---- Draw final canvas ----
        full = self.patch_size * self.grid_size
        canvas = Image.new("RGB", (full, full), (255, 255, 255))

        for shape_name, i, j in chosen_objects:
            obj_img = self.draw_shape_in_patch(shape_name)
            canvas.paste(obj_img, (j * self.patch_size, i * self.patch_size))

        return canvas, chosen_objects

ALL_SHAPES = {
    "red_square", "green_square", "blue_square", "yellow_square",
    "red_circle", "green_circle", "blue_circle", "yellow_circle",
    "red_triangle", "green_triangle", "blue_triangle", "yellow_triangle",
    "red_star", "green_star", "blue_star", "yellow_star"
}

if __name__ == "__main__":
    # Clean old data
    if os.path.exists("data") and os.path.isdir("data"):
        shutil.rmtree("data")

    os.makedirs("data/images", exist_ok=True)

    random.seed(42)

    GRID_SIZE = 12
    NUM_OBJECTS = 15
    MANHATTAN_DIST = 2
    TARGET_SHAPE = "green_circle"
    NUMBER_OF_PAIRS = 200  # <-- choose how many pairs you want

    # Allowed shapes: include TARGET_SHAPE plus any non-target shapes you like
    
    allowed_shapes = [
    # "red_square", "blue_square", "yellow_square",
    # "green_circle",
    # "red_triangle", "blue_triangle", "yellow_triangle",
    # "red_star", "blue_star", "yellow_star"
    "red_circle", "green_circle", "blue_circle", "yellow_circle",
    ]

    non_target_shapes = [
    # "red_square", "blue_square", "yellow_square",
    # "red_triangle", "blue_triangle", "yellow_triangle",
    # "red_star", "blue_star", "yellow_star"
    "red_circle", "blue_circle", "yellow_circle",
    ]
    


    # Generator that produces exactly ONE target per image
    gen_with_target = ObjectGenerator(
        patch_size=28,
        grid_size=GRID_SIZE,
        min_manhattan_dist=MANHATTAN_DIST,
        allowed_shapes=allowed_shapes,
        max_counts={TARGET_SHAPE: 1},   # at most 1 target
        min_counts={TARGET_SHAPE: 1},   # at least 1 target
    )

    with open("data/annotations.jsonl", "w", encoding="utf-8") as ann_file:
        for pair_id in range(NUMBER_OF_PAIRS):

            # ----- 1) Generate an image WITH target, interior (for 3x3 neighborhood) -----
            while True:
                img_with, placed = gen_with_target.generate(num_objects=NUM_OBJECTS)
                # find the single target
                target_list = [(s, r, c) for (s, r, c) in placed if s == TARGET_SHAPE]
                assert len(target_list) == 1, "Generator did not produce exactly one target"
                _, tr, tc = target_list[0]

                # ensure target is not on the border so we have full 3x3 neighborhood
                if 1 <= tr <= GRID_SIZE - 2 and 1 <= tc <= GRID_SIZE - 2:
                    break  # accept this sample; otherwise regenerate

            # ----- 2) Build matching image WITHOUT target (same layout, different shape) -----
            # Replace the target shape with some non-target shape in the same cell
            replacement_candidates = non_target_shapes  # anything except TARGET_SHAPE

            placed_without = []
            for (shape_name, r, c) in placed:
                if shape_name == TARGET_SHAPE:
                    new_shape = random.choice(replacement_candidates)
                    placed_without.append((new_shape, r, c))
                else:
                    placed_without.append((shape_name, r, c))

            # Draw the "without target" image manually
            full = gen_with_target.patch_size * gen_with_target.grid_size
            img_without = Image.new("RGB", (full, full), (255, 255, 255))
            for shape_name, r, c in placed_without:
                obj_img = gen_with_target.draw_shape_in_patch(shape_name)
                img_without.paste(obj_img, (c * gen_with_target.patch_size,
                                            r * gen_with_target.patch_size))

            # ----- 3) Save both images -----
            fname_with = f"data/images/pair_{pair_id:05d}_with.png"
            fname_without = f"data/images/pair_{pair_id:05d}_without.png"
            img_with.save(fname_with)
            img_without.save(fname_without)

            # ----- 4) Write annotations (include pair info + target coords) -----
            ann_with = gen_with_target.annotation_dict(fname_with, placed)
            ann_with["pair_id"] = pair_id
            ann_with["has_target"] = True
            ann_with["target"] = {
                "shape": "circle",
                "color": "green",
                "r": tr,
                "c": tc,
            }

            ann_without = gen_with_target.annotation_dict(fname_without, placed_without)
            ann_without["pair_id"] = pair_id
            ann_without["has_target"] = False
            ann_without["target"] = {
                "shape": "circle",
                "color": "green",
                "r": tr,
                "c": tc,
            }

            ann_file.write(json.dumps(ann_with) + "\n")
            ann_file.write(json.dumps(ann_without) + "\n")
