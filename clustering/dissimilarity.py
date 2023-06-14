import numpy as np
import pandas as pd
from math import *
from decimal import Decimal
from scipy.spatial import distance
import gower

# ======================================================================================================================
# Function distance between two points
# and calculate distance value to given
# root value(p is root value)
def p_root(value, root):
    root_value = 1 / float(root)
    return round(Decimal(value) **
                 Decimal(root_value), 3)


def minkowski_distance(x, y, p_value):
    # pass the p_root function to calculate
    # all the value of vector parallelly
    return (p_root(sum(pow(abs(a - b), p_value)
                       for a, b in zip(x, y)), p_value))


def calcMinkowskiDist(vector1, vector2):
    print('from customised function')
    print(minkowski_distance(x=vector1, y=vector2, p_value=3))
    print('\nfrom scipy')
    print(distance.minkowski(u=vector1, v=vector2, p=3))


# ======================================================================================================================
def calcGowerDist(dfTrips):
    """
    calculates the matrix of Gower distance between all pais of individuals
    :param dfTrips: dataframe of trip attributes without one-hot-encoding catageorical features.
    :return:
    """
    distanceMatrix = gower.gower_matrix(dfTrips)
    return distanceMatrix


def calcGowerDist1Pair(x, y):
    xy = np.array([x, y])
    df = pd.DataFrame(xy)
    gowerMat = gower.gower_matrix(df)
    print(gowerMat)
    gowerDist = gowerMat[0][1]
    return gowerDist
