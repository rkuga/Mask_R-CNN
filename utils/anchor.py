import numpy as np
import random
import cv2


def generate_anchors(base_size=15, ratios=(0.5, 1, 2), scales=(8, 16, 32)):
    # Generate anchor (reference) windows by enumerating aspect ratios X
    # scales w.r.t. a reference (0, 0, 15, 15) window.
    base_anchor = np.array([0, 0, base_size, base_size])
    ratio_anchors = _ratio_enum(base_anchor, np.asarray(ratios))
    anchors = np.vstack(
        [_scale_enum(ratio_anchors[i, :], np.asarray(scales))+1
         for i in range(len(ratio_anchors))])
    return anchors

def _ratio_enum(anchor, ratios):
    # Enumerate a set of anchors for each aspect ratio wrt an anchor.
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.rint(np.sqrt(size_ratios))
    hs = np.rint(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    # Enumerate a set of anchors for each scale wrt an anchor.
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _mkanchors(ws, hs, x_ctr, y_ctr):
    # Given a vector of widths (ws) and heights (hs) around a center
    # (x_ctr, y_ctr), output a set of anchors (windows).
    ws, hs = ws[:, None], hs[:, None]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)))
    return anchors

def _whctrs(anchor):
    # Return width, height, x center, and y center for an anchor (window).
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr