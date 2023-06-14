import pandas as pd
import numpy as np
import time
import pickle

from sklearn.cluster import KMeans
from sklearn.cluster import BisectingKMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import euclidean_distances

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import clusterClassifier
import visualisation

# ======================================================================================================================
def findClustersBisectingKmeans(dfTrips, clusteringParams, kwargs):
    tripCols = dfTrips.columns.tolist()
    if 'indivID' in tripCols:
        tripCols.remove('indivID')

    dfClusteringResults = pd.DataFrame()
    for nClusters in clusteringParams['n_clusters']:
        for initVal in clusteringParams['init']:
            for n_initVal in clusteringParams['n_init']:
                for bisectingStrat in clusteringParams['bisecting_strategy']:
                    tik = time.time()
                    clstrAlgo = BisectingKMeans(n_clusters=nClusters, init=initVal, n_init=n_initVal,
                                                bisecting_strategy=bisectingStrat, **kwargs)
                    clstrAlgo.fit(dfTrips[tripCols])
                    silhouetteScore = silhouette_score(dfTrips[tripCols], clstrAlgo.labels_)
                    chScore = calinski_harabasz_score(dfTrips[tripCols], clstrAlgo.labels_)
                    dbScore = davies_bouldin_score(dfTrips[tripCols], clstrAlgo.labels_)
                    dfResults = pd.DataFrame.from_dict({'n_clusters': [nClusters],
                                                        'init': [initVal],
                                                        'n_init': [n_initVal],
                                                        'bisecting_strategy': [bisectingStrat],
                                                        'silhouetteScore': [silhouetteScore],
                                                        'chScore': [chScore],
                                                        'dbScore': [dbScore],
                                                        'clusterLabels': [clstrAlgo.labels_]})
                    dfClusteringResults = pd.concat([dfClusteringResults, dfResults])
                    print('\tbisectingK-means nClusters %d, initVal %s, n_initVal %d, strategy %s -- took %.2f secs' %
                          (nClusters, initVal, n_initVal, bisectingStrat, time.time()-tik))

    return dfClusteringResults

# ======================================================================================================================
def assignToNewCluster(aRow, dfTrips, tripAttribCols):
    if aRow['silScore'] >= 0:
        return aRow['cluster']
    # gets the list of clusters that aRow doesn't belong to
    otherClusters = dfTrips['cluster'].loc[dfTrips['cluster'] != aRow['cluster']].unique()
    npRow = aRow[tripAttribCols].to_frame().transpose().to_numpy()
    avgDists2OtherClusters = []
    for aCluster in otherClusters:
        npTripsSub = dfTrips.loc[dfTrips['cluster'] == aCluster][tripAttribCols].to_numpy()
        avgEuclDist = np.mean(euclidean_distances(npTripsSub, npRow))
        avgDists2OtherClusters.append(avgEuclDist)
    # gets index of the min distance in avgDists2OtherClusters
    idxMinDist = np.argmin(np.array(avgDists2OtherClusters))
    # and uses that index to get the corresponding cluster in otherClusters
    nearestCluster = otherClusters[idxMinDist]
    #print((aRow['cluster'], aRow['silScore'], otherClusters, avgDists2OtherClusters, nearestCluster))
    return nearestCluster

def postprocBisectingKmeans(dfTrips, dfClustering1Run):
    dfClustering1Run.reset_index(inplace=True, drop=True)
    tmpCols = ['nMinsTrip1', 'nMinsDepTimeTrip1', 'clusterLabels', 'silScore']
    counts = 0
    tripAttribCols = dfTrips.columns
    for index, row in dfClustering1Run.iterrows():
        if row['n_clusters'] == 2:
            continue
        # assigns cluster labels to
        dfTrips['cluster'] = row['clusterLabels']
        # calculates silhouette score for each sample
        dfTrips['silScore'] = silhouette_samples(dfTrips[tripAttribCols], row['clusterLabels'])
        # determines new cluster for rows with negative silhouette score
        dfTrips['newCluster'] = dfTrips.apply(lambda aRow: assignToNewCluster(aRow, dfTrips, tripAttribCols), axis=1)
        dfTrips['newSilScore'] = silhouette_samples(dfTrips[tripAttribCols], dfTrips['newCluster'])
        dfTrips.to_csv('dfTrips_newCluster.csv', index=True)
        dfTrips.loc[dfTrips['silScore'] < 0].to_csv('dfTrips_newCluster_negSil.csv', index=True)

        counts += 1
        if counts == 1:
            break


# ======================================================================================================================
# DEPRECATED ===========================================================================================================
# ======================================================================================================================
def findClusters(dfTrips, algoStr):
    tripCols = dfTrips.columns.tolist()
    if 'indivID' in tripCols:
        tripCols.remove('indivID')
    #tripsArray = dfTrips[tripCols].to_numpy()

    silhouetteCoeff = []
    chIndex = []
    nClusters = []
    # density based methods
    if algoStr == 'DBSCAN':
        epsList = np.arange(0.01, 1, 0.01).tolist()  # epsList = np.arange(0.001, .01, 0.001).tolist()
        silScores, chScores, nClusters, clusterLabels = runDBSCAN(epsList, dfTrips[tripCols])
        return silScores, chScores, nClusters, clusterLabels
    elif algoStr == 'OPTICS':
        maxEpsList = np.arange(5, 10, 1).tolist()
        silScores, chScores, nClusters, clusterLabels = runOPTICS(maxEpsList, dfTrips[tripCols])
        return silScores, chScores, nClusters, clusterLabels
    # hierarchical methods
    elif algoStr == 'Agglomerative':
        numbersOfClusters = np.arange(5, 10, 5).tolist()  # [i for i in range(10, 100)]
        silScores, chScores, nClusters, clusterLabels = runAgglomerative(numbersOfClusters, dfTrips[tripCols])
        return silScores, chScores, nClusters, clusterLabels
    # centroid based methods (partitional clustering)
    elif algoStr == 'Birch':
        thresholdList = np.arange(.1, 1, .1).tolist()
        silScores, chScores, nClusters, clusterLabels = runBirch(thresholdList, dfTrips[tripCols])
        return silScores, chScores, nClusters, clusterLabels
    elif algoStr == 'Kmeans':
        numbersOfClusters = np.arange(5, 10, 5).tolist()  # [i for i in range(10, 100)]
        silScores, chScores, nClusters, clusterLabels = runKmeans(numbersOfClusters, dfTrips[tripCols])
        return silScores, chScores, nClusters, clusterLabels


# ======================================================================================================================
def runDBSCAN(epsList, dfTripsOnly):
    #tripsArray = dfTripsOnly.to_numpy()
    silhouetteCoeff = []
    chIndex = []
    nClusters = []
    clusterLabels = {}
    for eps in epsList:
        tik = time.time()
        clstrAlgo = DBSCAN(eps=eps)
        clstrAlgo.fit(dfTripsOnly)
        # updates output variables with clustering results
        clusterLabels[eps] = clstrAlgo.labels_
        silCoeffVal = silhouette_score(dfTripsOnly, clstrAlgo.labels_)
        chIndexVal = calinski_harabasz_score(dfTripsOnly, clstrAlgo.labels_)
        uLabels = list(set(clstrAlgo.labels_.tolist()))
        silhouetteCoeff.append(silCoeffVal)
        chIndex.append(chIndexVal)
        nClusters.append(len(uLabels))

        print('\teps %.3f, nClusters %d, silCoeffVal %.3f, chIndexVal %.3f, completed in %.2f secs' %
              (eps, len(uLabels), silCoeffVal, chIndexVal, (time.time() - tik)))

    return silhouetteCoeff, chIndex, nClusters, clusterLabels

# ======================================================================================================================
def runOPTICS(maxEpsList, dfTripsOnly):
    # tripsArray = dfTripsOnly.to_numpy()
    silhouetteCoeff = []
    chIndex = []
    nClusters = []
    clusterLabels = {}
    for maxEps in maxEpsList:
        tik = time.time()
        clstrAlgo = OPTICS(max_eps=maxEps, metric="minkowski", p=1, cluster_method="xi")
        clstrAlgo.fit(dfTripsOnly)
        # updates output variables with clustering results
        clusterLabels[maxEps] = clstrAlgo.labels_
        silCoeffVal = silhouette_score(dfTripsOnly, clstrAlgo.labels_)
        chIndexVal = calinski_harabasz_score(dfTripsOnly, clstrAlgo.labels_)
        uLabels = list(set(clstrAlgo.labels_.tolist()))
        silhouetteCoeff.append(silCoeffVal)
        chIndex.append(chIndexVal)
        nClusters.append(len(uLabels))

        print('\tmaxEps %.3f, nClusters %d, silCoeffVal %.3f, chIndexVal %.3f, completed in %.2f secs' %
              (maxEps, len(uLabels), silCoeffVal, chIndexVal, (time.time() - tik)))

    return silhouetteCoeff, chIndex, nClusters, clusterLabels

# ======================================================================================================================
def runAgglomerative(numbersOfClusters, dfTripsOnly):
    # tripsArray = dfTripsOnly.to_numpy()
    silhouetteCoeff = []
    chIndex = []
    nClusters = []
    clusterLabels = {}
    for numOfClusters in numbersOfClusters:
        tik = time.time()
        clstrAlgo = AgglomerativeClustering(n_clusters=numOfClusters, affinity="euclidean", linkage='ward')
        clstrAlgo.fit(dfTripsOnly)
        # updates output variables with clustering results
        clusterLabels[numOfClusters] = clstrAlgo.labels_
        silCoeffVal = silhouette_score(dfTripsOnly, clstrAlgo.labels_)
        chIndexVal = calinski_harabasz_score(dfTripsOnly, clstrAlgo.labels_)
        uLabels = list(set(clstrAlgo.labels_.tolist()))
        silhouetteCoeff.append(silCoeffVal)
        chIndex.append(chIndexVal)
        nClusters.append(len(uLabels))

        print('\tnumOfClusters %d, nClusters %d, silCoeffVal %.3f, chIndexVal %.3f, completed in %.2f secs' %
              (numOfClusters, len(uLabels), silCoeffVal, chIndexVal, (time.time() - tik)))

    return silhouetteCoeff, chIndex, nClusters, clusterLabels

# ======================================================================================================================
def runKmeans(numbersOfClusters, dfTripsOnly):
    # tripsArray = dfTripsOnly.to_numpy()
    silhouetteCoeff = []
    chIndex = []
    nClusters = []
    clusterLabels = {}
    for numOfClusters in numbersOfClusters:
        tik = time.time()
        kmeans_kwargs = {'init': 'random', 'n_init': 10, 'max_iter': 300, 'random_state': 2911}
        clstrAlgo = KMeans(n_clusters=numOfClusters, **kmeans_kwargs)
        clstrAlgo.fit(dfTripsOnly)
        # updates output variables with clustering results
        clusterLabels[numOfClusters] = clstrAlgo.labels_
        silCoeffVal = silhouette_score(dfTripsOnly, clstrAlgo.labels_)
        chIndexVal = calinski_harabasz_score(dfTripsOnly, clstrAlgo.labels_)
        uLabels = list(set(clstrAlgo.labels_.tolist()))
        silhouetteCoeff.append(silCoeffVal)
        chIndex.append(chIndexVal)
        nClusters.append(len(uLabels))

        print('\tnumOfClusters %d, nClusters %d, silCoeffVal %.3f, chIndexVal %.3f, completed in %.2f secs' %
              (numOfClusters, len(uLabels), silCoeffVal, chIndexVal, (time.time() - tik)))

    return silhouetteCoeff, chIndex, nClusters, clusterLabels

# ======================================================================================================================
def runBirch(thresholdList, dfTripsOnly):
    # tripsArray = dfTripsOnly.to_numpy()
    silhouetteCoeff = []
    chIndex = []
    nClusters = []
    clusterLabels = {}
    for threshold in thresholdList:
        tik = time.time()
        clstrAlgo = Birch(threshold=threshold, branching_factor=50)
        clstrAlgo.fit(dfTripsOnly)
        # updates output variables with clustering results
        clusterLabels[threshold] = clstrAlgo.labels_
        silCoeffVal = silhouette_score(dfTripsOnly, clstrAlgo.labels_)
        chIndexVal = calinski_harabasz_score(dfTripsOnly, clstrAlgo.labels_)
        uLabels = list(set(clstrAlgo.labels_.tolist()))
        silhouetteCoeff.append(silCoeffVal)
        chIndex.append(chIndexVal)
        nClusters.append(len(uLabels))

        print('\tthreshold %d, nClusters %d, silCoeffVal %.3f, chIndexVal %.3f, completed in %.2f secs' %
              (threshold, len(uLabels), silCoeffVal, chIndexVal, (time.time() - tik)))

    return silhouetteCoeff, chIndex, nClusters, clusterLabels
