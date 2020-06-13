import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from raster_to_svg  import graph_seg
from raster_to_svg import render_svg
from raster_to_svg import save_svg
from raster_to_svg import to_svg
img = imread("flower.jpg")
if img.shape[2] == 4:
    img = img[:,:,0:3]
seg = graph_seg(
    img,
    thresh=80,
)
svg = to_svg(img, seg)
img2 = render_svg(svg)
fig = plt.subplots(nrows=1, ncols=2)
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(img2)
plt.show()
save_svg(svg, "out.svg")
