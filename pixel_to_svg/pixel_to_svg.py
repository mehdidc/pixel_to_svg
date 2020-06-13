import uuid
import os
from collections import namedtuple
from subprocess import call
from io import StringIO, BytesIO

import numpy as np
from imageio import imread, imsave
from skimage import segmentation
from skimage.future import graph

from svgpathtools import svg2paths2
from svgpathtools import disvg, wsvg
from cairosvg import svg2png

SVG = namedtuple("SVG", "paths attributes")

def to_svg(img, seg):
    nb = seg.max() + 1
    P = []
    A = []
    for layer in range(nb):
        mask = (seg == layer)
        if np.all(mask==0):
            continue
        paths, attrs, svg_attrs = binary_image_to_svg2(mask)
        for attr in attrs:
            r, g, b, *rest = img[mask].mean(axis=0)
            r = int(r)
            g = int(g)
            b = int(b)
            col = f"rgb({r},{g},{b})"
            attr["stroke"] = col
            attr["fill"] = col
        P.extend(paths)
        A.extend(attrs)
    return SVG(paths=P, attributes=A)

def render_svg(svg, width=None, height=None):
    drawing = wsvg(
        paths=svg.paths, 
        attributes=svg.attributes,
        paths2Drawing=True,
    )
    fd = StringIO()
    drawing.write(fd)
    fo = BytesIO()
    svg2png(bytestring=fd.getvalue(), write_to=fo, output_width=width, output_height=height)
    fo.seek(0)
    return imread(fo, format="png")


def wsvg(paths=None, colors=None,
          filename=os.path.join(os.getcwd(), 'disvg_output.svg'),
          stroke_widths=None, nodes=None, node_colors=None, node_radii=None,
          openinbrowser=False, timestamp=False,
          margin_size=0.1, mindim=600, dimensions=None,
          viewbox=None, text=None, text_path=None, font_size=None,
          attributes=None, svg_attributes=None, svgwrite_debug=False, paths2Drawing=False):
    """Convenience function; identical to disvg() except that
    openinbrowser=False by default.  See disvg() docstring for more info."""
    return disvg(paths, colors=colors, filename=filename,
          stroke_widths=stroke_widths, nodes=nodes,
          node_colors=node_colors, node_radii=node_radii,
          openinbrowser=openinbrowser, timestamp=timestamp,
          margin_size=margin_size, mindim=mindim, dimensions=dimensions,
          viewbox=viewbox, text=text, text_path=text_path, font_size=font_size,
          attributes=attributes, svg_attributes=svg_attributes,
          svgwrite_debug=svgwrite_debug, paths2Drawing=paths2Drawing)

def save_svg(svg, out="output.svg"):
    wsvg(
        paths=svg.paths, 
        attributes=svg.attributes,
        filename=out,
    )

def binary_image_to_svg(seg):
    seg = (1-seg)
    seg = (seg*255).astype("uint8")
    seg = seg[::-1]
    name = str(uuid.uuid4())
    bmp = name + ".bmp"
    svg = name + ".svg"
    imsave(bmp, seg)
    call(f"potrace -s {bmp}", shell=True)
    paths = svg2paths2(svg)
    os.remove(bmp)
    os.remove(svg)
    return paths

def binary_image_to_svg2(mask):
    import potrace
    bmp = potrace.Bitmap(mask)
    bmp.trace()
    xml = bmp.to_xml()
    fo = StringIO()
    fo.write(xml)
    fo.seek(0)
    paths = svg2paths2(fo)
    return paths



def graph_seg(img, max_dist=200, thresh=80, sigma=255.0):
    img = img.astype("float")
    seg = segmentation.quickshift(
        img,
        max_dist=max_dist,
    )
    g = graph.rag_mean_color(
        img, 
        seg,
        sigma=sigma,
    )
    seg = graph.merge_hierarchical(
        seg, 
        g, 
        thresh=thresh, 
        rag_copy=False,
        in_place_merge=True,
        merge_func=_merge_mean_color,
        weight_func=_weight_mean_color
    )
    return seg


def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def _merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skimage.transform import resize
    img = imread("bird.png").astype("float")
    img = resize(img, (256,256), preserve_range=True)
    img = img[:,:,0:3]
    plt.imshow(img/255)
    plt.show()
    seg = graph_seg(img)
    plt.imshow(seg, cmap="tab20c")
    plt.show()
    svg = to_svg(img, seg)
    # save_svg(svg, out="out.svg")
    img = render_svg(svg)
    plt.imshow(img)
    plt.show()
