import numpy as np
import pandas as pd
import pickle

import visualisation
import dunnIndex

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics.pairwise import euclidean_distances


# ======================================================================================================================
def mkClusterLabelsDataFrame(df_clustering_results):
    newDict = {}
    for index, row in df_clustering_results.iterrows():
        nClusters = row['n_clusters']
        init = row['init']
        n_init = row['n_init']
        bisectStrat = row['bisecting_strategy']
        clusterLabels = row['clusterLabels']
        newDict['%d_%s_%d_%s' % (nClusters, init, n_init, bisectStrat)] = clusterLabels
    df_cluster_labels = pd.DataFrame.from_dict(newDict)
    return df_cluster_labels


# ======================================================================================================================
def calcDunnIndex(df_clustering_results, dfTrips_ohe_std):
    counts = 0
    for index, row in df_clustering_results.iterrows():
        nClusters = row['n_clusters']
        init = row['init']
        n_init = row['n_init']
        bisectStrat = row['bisecting_strategy']
        clusterLabels = row['clusterLabels']
        dunc = dunnIndex.dunn(clusterLabels,
                              euclidean_distances(dfTrips_ohe_std.to_numpy()))
        print('nClusters %d, init %s, n_init %d, bisectStrat %s -- dunnIndex %.6f' %
              (nClusters, init, n_init, bisectStrat, dunc))
        counts += 1
        if counts == 10:
            break


# ======================================================================================================================
if __name__ == '__main__':
    '''
    # make plots of CH and DB scores versus number of clusters from all runs
    pklPath = '../myPy/tmpOutputs/BisectingKmeans'
    pklFilenames = {'%s/dfClusteringResults_rand2911_v2.pkl' % pklPath: 'Run 1',
                    '%s/dfClusteringResults_rand2910_v2.pkl' % pklPath: 'Run 2',
                    '%s/dfClusteringResults_rand2611_v2.pkl' % pklPath: 'Run 3',
                    '%s/dfClusteringResults_rand1103_v2.pkl' % pklPath: 'Run 4',
                    '%s/dfClusteringResults_rand311_v2.pkl' % pklPath: 'Run 5',
                    '%s/dfClusteringResults_rand409_v2.pkl' % pklPath: 'Run 6',
                    '%s/dfClusteringResults_rand2212_v2.pkl' % pklPath: 'Run 7',
                    '%s/dfClusteringResults_rand111_v2.pkl' % pklPath: 'Run 8',
                    '%s/dfClusteringResults_rand711_v2.pkl' % pklPath: 'Run 9'}
    dfResultsMultiRuns = pd.DataFrame()
    for pklFile, run in pklFilenames.items():
        dfTmp = pd.read_pickle(pklFile)
        # we use only clustering results correponding to n_init=19.
        # all runs use only n_init=19, except for run 2911 which has results for n_init values of 1 and 19.
        dfTmp = dfTmp.loc[dfTmp['n_init'] == dfTmp['n_init'].max()]
        dfTmp['run'] = run
        dfResultsMultiRuns = pd.concat([dfResultsMultiRuns, dfTmp])
    visualisation.plot_CH_DB_multiRuns(dfResultsMultiRuns, '../myPy/plots/CH_DB_multiRuns.html')
    # make plots of average Silhouette score and boxplot of cluster Silhouette scores versus number of clusters
    # results in a 3x3 plot matrix (for 9 runs)
    visualisation.plotAvgSilhouetteMultiRuns(pklFilenames, '../myPy/plots/silhouette_multiRuns.html')
    visualisation.plotClusterSizesMultiRuns(pklFilenames, '../myPy/plots/clusterSizes_multiRuns.html')
    visualisation.plotStdevClsizesVersusMedSilScores(pklFilenames, '../myPy/plots/stdevPareto_multiRuns.html')
    visualisation.plotLoglossVersusClSizes(pd.read_csv('../myPy/logLossSummaryOfRuns.csv'),
                                           pklFilenames, '../myPy/plots/loglossCV_multiRuns.html')
    '''

    # makes a horizontal bar plot of variable importance for a given nClusters from a given runId.
    '''
    # plots features most important to the clustering for number of clusters <= 10 for each run
    pklPath = '../myPy/tmpOutputs/BisectingKmeans'
    pklFilenames = {'rand2911': ['Run 1', [2, 8, 9, 10]],
                    'rand2910': ['Run 2', [2, 9, 10]],
                    'rand2611': ['Run 3', [2, 8, 9, 10]],
                    'rand1103': ['Run 4', [2, 8, 9, 10]],
                    'rand311': ['Run 5', [2, 8, 9, 10]],
                    'rand409': ['Run 6', [2, 8, 9]],
                    'rand2212': ['Run 7', [2, 8, 9, 10]],
                    'rand111': ['Run 8', [2, 8, 9, 10]],
                    'rand711': ['Run 9', [2, 8, 9, 10]]}
    nBiggestClusters = 3
    nTripAttribs2Plot = 4
    for runId, runDetails in pklFilenames.items():
        runDesc = runDetails[0]
        nClustersList = runDetails[1]
        for nClusters in nClustersList:
            print('%s, %d clusters' % (runId, nClusters))
            col2Plot = 'scaled_importance'
            dfImportance = pd.read_csv('../myPy/tmpOutputs/drfVarImportance/%s/%dClusters.csv' % (runId, nClusters))
            visualisation.plotVarImportance(dfImportance, runId, nClusters, col2Plot,
                                            '../myPy/plots/varImportance_%s/%d_Clusters.html' % (runDesc, nClusters))
            # plots trip attributes that are most inflential to clustering of the biggest clusters
            # defines trip attributes that are most influential to clustering (from the trained RF classifier)
            # (these attributes are decided from the above plot of variable importance)
            dfImportance.sort_values(by=[col2Plot], inplace=True, ascending=False)
            vars2Plot = dfImportance.iloc[:nTripAttribs2Plot]['variable'].values.tolist()
            # reads in the attribute of all trips and the clustering results
            dfClustering1Run = pd.read_pickle(
                '../myPy/tmpOutputs/BisectingKmeans/dfClusteringResults_%s_v2.pkl' % runId)
            dfTrips = pd.read_pickle('../myPy/tmpOutputs/dfTrips.pkl')  # reads in the attribute of all trips
            # plots these attributes for the nBiggestClusters
            outFilename = '../myPy/plots/tripVars_%s/%dCls_%dBiggest.html' % (runDesc, nClusters, nBiggestClusters)
            visualisation.plotTripVarsBiggestClusters(dfTrips, dfClustering1Run, nClusters, nBiggestClusters,
                                                      vars2Plot, runDesc, outFilename)
    
    # plots demographic attributes of people in each of the nBiggestClusters
    for runId, runDetails in pklFilenames.items():
        runDesc = runDetails[0]
        nClustersList = runDetails[1]
        for nClusters in nClustersList:
            # reads in HTS data for demographics attibutes (after the preprocessing of these demographics attribs)
            dfHTS = pd.read_csv('../myPy/tmpOutputs/dfHTS_tripProcessed3.csv')
            demoCols = ['age', 'gender', 'eduLevel', 'licence', 'jobType', 'employStatus',
                        'monthlyIncome', 'ownMotorVehicles']
            # reads in the attribute of all trips and the clustering results
            dfClustering1Run = pd.read_pickle(
                '../myPy/tmpOutputs/BisectingKmeans/dfClusteringResults_%s_v2.pkl' % runId)
            # plots these attributes for the nBiggestClusters
            outFilename = '../myPy/plots/demoVars_%s/%dCls_%dBiggest.html' % (runDesc, nClusters, nBiggestClusters)
            visualisation.plotDemoAttribsBiggestClusters(dfHTS[demoCols], dfClustering1Run, nClusters,
                                                         nBiggestClusters, runDesc, outFilename)
    '''

    lastNumber = 9
    alist = np.arange(1, lastNumber+1, 1)
    print(alist)

    '''
    runId = 'rand2212'
    dfClustering1Run = pd.read_pickle('../myPy/tmpOutputs/BisectingKmeans/dfClusteringResults_%s_v2.pkl' % runId)
    dfClustersAttribs = pd.read_pickle('../myPy/tmpOutputs/BisectingKmeans/dfClustersAttribs_%s.pkl' % runId)
    print('dfClustering1Run')
    print(dfClustering1Run.shape)
    print(dfClustering1Run.columns)
    print(dfClustering1Run[['n_clusters', 'clSizes', 'clMedSilScores', 'sampleSilhouette']])
    print('dfClustersAttribs')
    print(dfClustersAttribs.shape)
    print(dfClustersAttribs.columns)
    '''

    # plots silhouette scores of samples from a given runId (random state)
    '''
    runId = 'rand2212'
    dfClustering1Run = pd.read_pickle('../myPy/tmpOutputs/BisectingKmeans/dfClusteringResults_%s_v2.pkl' % runId)
    #visualisation.plotSilhouetteScoreSamples(dfClustering1Run, runId)
    visualisation.plotClusteringEvalScores1Run(dfClustering1Run, '../myPy/plots/clusteringEvalv2_%s.html'%runId, runId)
    '''

    # make boxplots of evaluation metrics of clustring results from multiple runs
    '''
    pklFilenames = ['dfClusteringResults_rand2911.pkl', 'dfClusteringResults_rand2910.pkl',
                    'dfClusteringResults_rand2611.pkl', 'dfClusteringResults_rand1103.pkl',
                    'dfClusteringResults_rand311.pkl', 'dfClusteringResults_rand409.pkl']
    dfResultsMultiRuns = pd.DataFrame()
    for pklFile in pklFilenames:
        dfTmp = pd.read_pickle('../myPy/tmpOutputs/BisectingKmeans/%s' % pklFile)
        # we use only clustering results correponding to n_init=19.
        # all runs use only n_init=19, except for run 2911 which has results for n_init values of 1 and 19.
        dfTmp = dfTmp.loc[dfTmp['n_init'] == dfTmp['n_init'].max()]
        dfResultsMultiRuns = pd.concat([dfResultsMultiRuns, dfTmp])
    visualisation.plotClusteringEvalScoresAllRuns(dfResultsMultiRuns, '../myPy/plots/clusteringEvalAllRuns.html')
    '''

    # consolidate clustering results from random_state 2911
    '''
    pklFilenames = ['dfClusteringResults_nCl2_50.pkl', 'dfClusteringResults_nCl52_60.pkl',
                    'dfClusteringResults_nCl62_70.pkl', 'dfClusteringResults_nCl72_100.pkl']
    dfClusteringResults = pd.DataFrame()
    for pklFile in pklFilenames:
        dfClusteringResults = pd.concat([dfClusteringResults,
                                         pd.read_pickle('../myPy/tmpOutputs/BisectingKmeans/%s' % pklFile)])
    dfClusteringResults.reset_index(inplace=True, drop=True)

    nClustersList2Plot = np.arange(2, 101, 2)
    n_inits2Plot = [1, 19]
    dfClusteringResults = dfClusteringResults.loc[(dfClusteringResults['n_clusters'].isin(nClustersList2Plot)) &
                                                  (dfClusteringResults['n_init'].isin(n_inits2Plot))]
    dfClusteringResults.to_pickle('../myPy/tmpOutputs/BisectingKmeans/dfClusteringResults_rand2911.pkl')
    '''

    # plot evaluation scores for 1 random state
    '''
    dfClusteringResults = pd.read_pickle('../myPy/tmpOutputs/BisectingKmeans/dfClusteringResults_rand2911.pkl')
    visualisation.plotClusteringEvalScores_v2(dfClusteringResults, '../myPy/plots/clusteringEval_rand2911.html')
    '''

    # prepare clustering labels for classification
    '''
    nClustersList = np.arange(2, 61, 2)
    dfClusteringResultsSub = dfClusteringResults.loc[dfClusteringResults['n_clusters'].isin(nClustersList)]
    dfClusterLabels = mkClusterLabelsDataFrame(dfClusteringResultsSub)
    dfClusterLabels.to_csv('./tmpOutputs/dfClusterLabels.csv', index=False)
    print(dfClusterLabels.head())
    '''

    # test calculating Dunn Index
    '''
    data = load_iris()
    c = data['target']
    x = data['data']
    k = KMeans(n_clusters=3).fit_predict(x)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(x)
    clLbls = kmeans.labels_
    #k = kmeans.predict(x)

    dunc = dunnIndex.dunn(c, euclidean_distances(x))
    dunk = dunnIndex.dunn(k, euclidean_distances(x))
    print(dunc)
    print(dunk)
    
    # calculate Dunn index for clustering results - out of memory
    dfResults = pd.read_pickle('../myPy/tmpOutputs/BisectingKmeans/dfClusteringResults_rand2911.pkl')
    dfTrips_ohe_std = pd.read_pickle('../myPy/tmpOutputs/dfTrips_ohe_std.pkl')
    calcDunnIndex(dfResults, dfTrips_ohe_std)
    '''