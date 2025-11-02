import numpy as np
from PIL import Image, ImageDraw

def heatmap_10x10(vec100, vmin=None, vmax=None, size=280):
    H = np.array(vec100, dtype=np.float64).reshape(10,10)
    if vmin is None: vmin = float(H.min())
    if vmax is None: vmax = float(H.max())
    norm = (H - vmin) / (vmax - vmin + 1e-12)
    img = (np.clip(norm, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(img, mode="L").resize((size,size), resample=Image.NEAREST).convert("RGBA")
    return img

def overlay_grid(img: Image.Image, grid=10, color=(0,0,0,128)):
    draw = ImageDraw.Draw(img)
    step = img.width // grid
    for i in range(1,grid):
        draw.line([(i*step,0),(i*step,img.height)], fill=color, width=1)
        draw.line([(0,i*step),(img.width,i*step)], fill=color, width=1)
    return img

def overlay_patches(img: Image.Image, cells, outline=(0,255,0,255)):
    draw = ImageDraw.Draw(img)
    grid = 10
    step = img.width // grid
    for r,c in cells:
        x0,y0 = c*step, r*step
        draw.rectangle([x0,y0,x0+step,y0+step], outline=outline, width=3)
    return img
