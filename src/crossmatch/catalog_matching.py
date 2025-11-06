"""
Catalog cross-matching using astropy for astronomical coordinate matching.

This module provides functions to match objects from different astronomical surveys
based on their sky coordinates.
"""

import numpy as np
import time
from astropy.coordinates import SkyCoord
from astropy import units as u


def crossmatch(cat1, cat2, max_dist):
    """
    Cross-match two astronomical catalogs based on sky coordinates.

    Parameters
    ----------
    cat1 : numpy.ndarray
        First catalog with shape (N, 2) containing [RA, Dec] in degrees
    cat2 : numpy.ndarray
        Second catalog with shape (M, 2) containing [RA, Dec] in degrees
    max_dist : float
        Maximum matching distance in degrees

    Returns
    -------
    matches : list of tuples
        List of (cat1_idx, cat2_idx, distance) for matched objects
    no_matches : list
        List of indices from cat1 that had no match in cat2
    time_taken : float
        Time taken to perform the cross-match in seconds
    """
    matches = []
    nomatches = []

    start = time.perf_counter()

    # Convert to SkyCoord objects
    skycat1 = SkyCoord(cat1*u.degree, frame='icrs')
    skycat2 = SkyCoord(cat2*u.degree, frame='icrs')

    # Perform the matching
    closest_ids, closest_dists, closest_dists3d = skycat1.match_to_catalog_sky(skycat2)
    closest_dists_deg = closest_dists.value

    # Filter by maximum distance
    for cat1idx in range(len(cat1)):
        if closest_dists_deg[cat1idx] > max_dist:
            nomatches.append(cat1idx)
        else:
            matches.append((cat1idx, closest_ids[cat1idx], closest_dists_deg[cat1idx]))

    return matches, nomatches, time.perf_counter() - start


def save_matches(matches, output_file):
    """
    Save match results to a text file.

    Parameters
    ----------
    matches : list of tuples
        List of (cat1_idx, cat2_idx, distance) for matched objects
    output_file : str
        Path to output file
    """
    with open(output_file, 'w') as f:
        for match in matches:
            line = ' '.join(str(x) for x in match)
            f.write(line + '\n')
