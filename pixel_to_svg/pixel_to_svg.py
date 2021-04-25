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

def to_svg(img, seg, nb_layers=None, palette=None, opacity=None):
    if len(seg.shape) == 2:
        if nb_layers is None:
            nb_layers = seg.max() + 1
        masks = np.zeros((seg.shape[0], seg.shape[1], nb_layers)).astype(bool)
        m = masks.reshape((-1,nb_layers))
        s = seg.reshape((-1,))
        m[np.arange(len(m)), s] = 1
        assert np.all(masks.argmax(axis=2) == seg)
    else:
        masks = seg
    P = []
    A = []
    for layer in range(masks.shape[2]):
        mask = masks[:,:,layer]
        if np.all(mask==0):
            continue
        paths, attrs, svg_attrs = binary_image_to_svg2(mask)
        for attr in attrs:
            if palette is None:
                r, g, b, *rest = img[mask].mean(axis=0)
            else:
                r, g, b = palette[layer]
            r = int(r)
            g = int(g)
            b = int(b)
            col = f"rgb({r},{g},{b})"
            attr["stroke"] = col
            attr["fill"] = col
            if opacity:
                attr["opacity"] = opacity[layer]
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
    #NB: this code is originally from <https://github.com/mathandy/svgpathtools>.
    # Thanks tho @mathandy
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
    """
    same as binary_image_to_svg, but use `pypotrace`
    instead of calling `potrace` from shell
    it is more convenient and faster, this way
    """
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
    """
    segment an image using quickshift and merge_hierarchical
    from scikit-image. In principle, any segmentation method
    can be used, this is just one example.
    """
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
    """
    NB: this code is originally from <https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_rag_merge.html>.
    Thanks to scikit-image authors.
    
    Callback to handle merging nodes by recomputing mean color.

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
    """
     NB: this code is originally from <https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_rag_merge.html>.
    Thanks to scikit-image authors.

    Callback called before merging two nodes of a mean color distance graph.

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

