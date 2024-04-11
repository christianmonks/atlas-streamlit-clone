import math
from PIL import Image

def add_image(name="front_page", scale=1):
    im = Image.open(f"./image/{name}.png")
    w, h = im.size
    return im.resize((int(math.floor(w * scale)), int(math.floor(h * scale))))
