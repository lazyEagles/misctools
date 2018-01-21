
"""Miscellaneous tools

"""

import csv
from typing import Tuple

import numpy as np

def create_recorders_from_csvfile(path: str) -> list:
    """Create recorders from a csv file

    """

    with open(path) as f:
        reader = csv.DictReader(f)
        recorders = list(reader)

    return recorders

Latlon = Tuple[np.ndarray, np.ndarray]

def calculate_distance(latlon1: Latlon, latlon2: Latlon) -> np.ndarray:
    """Calculate distance between two locations

    """

    phi1, phi2 = np.meshgrid(
        np.radians(latlon1[0]), np.radians(latlon2[0]), indexing="ij"
    )

    lamb1, lamb2 = np.meshgrid(
        np.radians(latlon1[1]), np.radians(latlon2[1]), indexing="ij"
    )

    delta_lamb = np.absolute(lamb1 - lamb2)
    
    delta_rho = np.arctan2(
        np.sqrt(
            np.square(np.cos(phi2)*np.sin(delta_lamb))
            + np.square(
                np.cos(phi1)*np.sin(phi2)
                - np.sin(phi1)*np.cos(phi2)*np.cos(delta_lamb)
            )
        ),
        np.sin(phi1)*np.sin(phi2) + np.cos(phi1)*np.cos(phi2)*np.cos(delta_lamb)
    )

    earth_radius = 6372795

    distance = earth_radius * delta_rho

    return distance
