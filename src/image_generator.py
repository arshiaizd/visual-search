import os
import shutil
import random
import json
from PIL import Image, ImageDraw

class ObjectGenerator:
    def __init__(self, patch_size=28, grid_size=10, min_manhattan_dist=1,
                 allowed_shapes=None, max_counts=None, min_counts=None):
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.min_manhattan_dist = min_manhattan_dist

        all_shapes = ["red_square", "green_square", "red_circle", "green_circle"]
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
        color = (255, 0, 0) if "red" in shape_name else (0, 255, 0)
        box = [cx-half, cy-half, cx+half, cy+half]
        if "square" in shape_name:
            draw.rectangle(box, fill=color)
        else:
            draw.ellipse(box, fill=color)
        return img

    def split_shape(self, shape_name):
        color = "red" if "red" in shape_name else "green"
        shape = "square" if "square" in shape_name else "circle"
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


if __name__ == "__main__":
    if os.path.exists("data") and os.path.isdir("data"):
        shutil.rmtree("data")


    os.makedirs("data/images", exist_ok=True)
    random.seed(42)  # optional reproducibility
    NUM_OBJECTS = 3
    MANHATTAN_DIST = 5
    GREEN_CIRCLE = 1
    OTHER_OBJECTS = 0
    NUMBER_OF_IMAGES = 50
    gen_with_green_circle = ObjectGenerator(
        min_manhattan_dist=MANHATTAN_DIST,
        allowed_shapes=["red_square","green_circle","red_circle","green_square"],
        max_counts={"red_circle": None, "green_circle": GREEN_CIRCLE, "red_square": None, "green_square": None},
        min_counts={"red_circle": 0, "green_circle": GREEN_CIRCLE, "red_square": 0, "green_square": 0}
    )
    gen_without_green_circle = ObjectGenerator(
        min_manhattan_dist=MANHATTAN_DIST,
        allowed_shapes=["red_square","green_circle","red_circle","green_square"],
        max_counts={"red_circle": None, "green_circle": 0, "red_square": None, "green_square": None},
        min_counts={"red_circle": OTHER_OBJECTS, "green_circle": 0, "red_square": OTHER_OBJECTS, "green_square": OTHER_OBJECTS}
    )
    with open("data/annotations.jsonl", "w", encoding="utf-8") as ann_file:
        for i in range(NUMBER_OF_IMAGES):
            img, placed = gen_with_green_circle.generate(num_objects=NUM_OBJECTS)
            fname = f"data/images/img_{i:05d}.png"
            img.save(fname)
            ann = gen_with_green_circle.annotation_dict(fname, placed)
            ann_file.write(json.dumps(ann) + "\n")
        for i in range(NUMBER_OF_IMAGES, 2 * NUMBER_OF_IMAGES):
            img, placed = gen_without_green_circle.generate(num_objects=NUM_OBJECTS)
            fname = f"data/images/img_{i:05d}.png"
            img.save(fname)
            ann = gen_without_green_circle.annotation_dict(fname, placed)
            ann_file.write(json.dumps(ann) + "\n")
