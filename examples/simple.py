import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from pixel_to_svg  import graph_seg
from pixel_to_svg import render_svg
from pixel_to_svg import save_svg
from pixel_to_svg import to_svg
img = imread("flower.jpg")
if img.shape[2] == 4:
    img = img[:,:,0:3]
    
# segmentation step.
# here each pixel is mapped to a category.
# this internally uses quickshift and hierarchical_merge
# from scikit-image (see the code for more info). 
# In principle, any segmentation method can be used
# Feel free to replace this with you preferred method.
seg = graph_seg(
    img,
    thresh=80,
)
# Given a segmented image, turn it to SVG.
# This internally uses `potrace`  for each
# pixel catgory obtained from segmentation
svg = to_svg(img, seg)

# Convert SVG back to raster to display it
# and compare it to original image
img2 = render_svg(svg)

fig = plt.subplots(nrows=1, ncols=2)
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(img2)
plt.show()

# save the SVG
save_svg(svg, "out.svg")
