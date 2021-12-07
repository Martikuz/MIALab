
# -*- coding: utf-8 -*-
"""
intensity_normalization.util.histogram_tools
Process the histograms of MR (brain) images
Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 01, 2021
"""



__all__ = [
    "get_first_tissue_mode",
    "get_largest_tissue_mode",
    "get_last_tissue_mode",
    "get_tissue_mode",
    "smooth_histogram",
]

from typing import Tuple

import numpy as np
import statsmodels.api as sm
from scipy.signal import argrelmax

PEAK = {
    "last": ["t1", "other", "last"],
    "largest": ["t2", "flair", "largest"],
    "first": ["pd", "md", "first"],
}
VALID_PEAKS = {m for modalities in PEAK.values() for m in modalities}
VALID_MODALITIES = VALID_PEAKS - {"last", "largest", "first"}

def smooth_histogram(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Use kernel density estimate to get smooth histogram
    Args:
        data: np.ndarray of image data
    Returns:
        grid: domain of the pdf
        pdf: kernel density estimate of the pdf of data
    """
    data_vec = data.flatten().astype(np.float64)
    bandwidth = data_vec.max() / 80  # noqa
    kde = sm.nonparametric.KDEUnivariate(data)
    kde.fit(kernel="gau", bw=bandwidth, gridsize=80, fft=True)
    pdf = 100.0 * kde.density
    grid = kde.support

    # print('*******IN smooth_histogram')

    return grid, pdf


def get_largest_tissue_mode(data: np.ndarray) -> float:
    """Mode of the largest tissue class
    Args:
        data: image data
    Returns:
        largest_tissue_mode (float): intensity of the mode
    """
    grid, pdf = smooth_histogram(data)
    largest_tissue_mode: float = grid[np.argmax(pdf)]

    # print('*******IN get_largest_tissue_mode')

    return largest_tissue_mode


def get_last_tissue_mode(
        data: np.ndarray,
        remove_tail: bool = True,
        tail_percentage: float = 96.0,
) -> float:
    """Mode of the highest-intensity tissue class
    Args:
        data: image data
        remove_tail: remove tail from histogram
        tail_percentage: if remove_tail, use the
            histogram below this percentage
    Returns:
        last_tissue_mode: mode of the highest-intensity tissue class
    """
    if remove_tail:
        threshold = np.percentile(data, tail_percentage)
        valid_mask = data <= threshold
        data = data[valid_mask]
    grid, pdf = smooth_histogram(data)
    maxima = argrelmax(pdf)[0]
    last_tissue_mode: float = grid[maxima[-1]]

    # print('*******IN get_last_tissue_mode')

    return last_tissue_mode


def get_first_tissue_mode(
        data: np.ndarray,
        remove_tail: bool = True,
        tail_percentage: float = 99.0,
) -> float:
    """Mode of the lowest-intensity tissue class
    Args:
        data: image data
        remove_tail: remove tail from histogram
        tail_percentage: if remove_tail, use the
            histogram below this percentage
    Returns:
        first_tissue_mode: mode of the lowest-intensity tissue class
    """
    if remove_tail:
        threshold = np.percentile(data, tail_percentage)
        valid_mask = data <= threshold
        data = data[valid_mask]
    grid, pdf = smooth_histogram(data)
    maxima = argrelmax(pdf)[0]
    first_tissue_mode: float = grid[maxima[0]]

    # print('*******IN get_first_tissue_mode')

    return first_tissue_mode


def get_tissue_mode(data: np.ndarray, modality: str) -> float:
    """Find the appropriate tissue mode given a modality"""
    modality_ = modality.lower()
    if modality_ in PEAK["last"]:
        mode = get_last_tissue_mode(data)
    elif modality_ in PEAK["largest"]:
        mode = get_largest_tissue_mode(data)
    elif modality_ in PEAK["first"]:
        mode = get_first_tissue_mode(data)
    else:
        modalities = ", ".join(VALID_PEAKS)
        msg = f"Modality {modality} not valid. Needs to be one of {modalities}."
        print(msg)

    # print('*******IN get_tissue_mode')

    return mode
