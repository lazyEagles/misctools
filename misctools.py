
"""Miscellaneous tools

"""

import csv
import functools
import operator
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

def calculate_path_delivery_probability(
    calculated_probs, path, link_probs, repeat
):
    """
    example:
       0.5      0.5      0.7
    0 -----> 1 -----> 2 ----> 3
    path = (0,1,2,3)
    link_probs = [1,0.5,0.5,0.7,0]

    """
    if repeat in calculated_probs[path]:
        return calculated_probs[path][repeat]

    if repeat < 1 or not path or len(set(path)) < len(path):
        return 0

    if len(path) == 1:
        calculated_probs[path][repeat] = 1.0
        return 1.0

    if repeat == 1:
        prob = functools.reduce(
            operator.mul, link_probs[:-1], 1
        )
        calculated_probs[path][repeat] = prob
        return prob

    probs = []
    for i in range(1,len(path)+1):
        prob1 = calculate_path_delivery_probability(
            calculated_probs,
            path[:i],
            link_probs[:i] + [0],
            1
        )
        prob2 = 1 - link_probs[i]
        prob3 = calculate_path_delivery_probability(
            calculated_probs,
            path[i-1:],
            [1] + link_probs[i:],
            repeat-1
        )
        probs.append(prob1*prob2*prob3)
    prob = sum(probs)
    calculated_probs[path][repeat] = prob

    return prob

