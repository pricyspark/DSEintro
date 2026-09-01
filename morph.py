import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def crop_to_aspect(img: MatLike, aspect: float) -> MatLike:
    h, w = img.shape[:2]
    curr_aspect = w / h

    if curr_aspect > aspect:
        new_w = max(1, round(h * aspect))
        x0 = (w - new_w) // 2
        return img[:, x0 : x0 + new_w]
    else:
        new_h = max(1, round(w / aspect))
        y0 = (h - new_h) // 2
        return img[y0 : y0 + new_h, :]


def dual_resize(
    source: MatLike,
    target: MatLike,
    n: int | float = 1,
    edge_length: int | None = None,
) -> tuple[MatLike, MatLike]:
    h1_og, w1_og = source.shape[:2]
    h2_og, w2_og = target.shape[:2]

    aspect1 = w1_og / h1_og
    aspect2 = w2_og / h2_og

    crop_aspect = (aspect1 * aspect2) ** 0.5

    source_crop = crop_to_aspect(source, crop_aspect)
    target_crop = crop_to_aspect(target, crop_aspect)

    crop_h = min(source_crop.shape[0], target_crop.shape[0])
    crop_w = round(crop_h * crop_aspect)

    if edge_length is not None:
        n = max(crop_h, crop_w) / edge_length

    output_width = max(1, round(crop_w / n))
    output_height = max(1, round(crop_h / n))

    source_crop = cv2.resize(
        source_crop,
        (output_width, output_height),
        interpolation=cv2.INTER_AREA,
    )

    target_crop = cv2.resize(
        target_crop,
        (output_width, output_height),
        interpolation=cv2.INTER_AREA,
    )

    return source_crop, target_crop


def morph(source: MatLike, target: MatLike) -> NDArray[np.uint8]:
    source_flat = source.reshape((-1, 3))
    target_flat = target.reshape((-1, 3))

    dist = cdist(target_flat, source_flat, metric="euclidean")
    _, col_idx = linear_sum_assignment(dist)

    morphed_flat = source_flat[col_idx]
    morphed = morphed_flat.reshape(source.shape).astype(np.uint8)

    return morphed
