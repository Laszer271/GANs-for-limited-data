from PIL import Image
import numpy as np

img = Image.open('profantasy/Angel.png')
bands = img.getbands()
arr = np.array(img)[12:25, 190:200]
mask = arr[..., -1] == 0

colors = arr[..., :-1][mask]
colors, cnts = np.unique(colors, axis=0, return_counts=True)

for color, cnt in zip(colors, cnts):
    print(color, '->', cnt)
