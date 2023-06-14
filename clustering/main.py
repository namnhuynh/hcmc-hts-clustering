import censusProc
import htsProc
import dimReduction
import clustering
import dissimilarity
import visualisation
import clusteringEval
import clusterClassifier

import numpy as np
import pickle
import time
import pandas as pd

pd.options.mode.chained_assignment = None

if __name__ == '__main__':
    '''
    print('read and process raw hts...')
    tik = time.time()
    dfHTS = htsProc.readHTS(readFromPkl=True)
    htsProc.processDemographicAttributes(dfHTS)
    htsProc.processTripAttributes(dfHTS)
    print(dfHTS.shape)
    print('\tcompleted preprocessing raw hts in %.2f seconds' % (time.time() - tik))
    '''

    ''' START IGNORED
    tik = time.time()
    colCorrelations, dfTripsFAMD = dimReduction.doFAMD(dfHTS)
    print('completed FAMD of trip attributes in %.2f seconds' % (time.time() - tik))
    dfTripsFAMD.to_csv('./tmpOutputs/famd/dfTrips_famd.csv', index=True)
    colCorrelations.to_csv('./tmpOutputs/famd/colCorrelations_famd.csv', index=False)

    tik = time.time()
    # applies DBSCAN clustering to dfTrips after FAMD
    dfTripsFAMD_DBSCAN = clustering.findClusters_DBSCAN(dfTripsFAMD)
    print('completed clustering (dbscan) of trip attributes in %.2f seconds' % (time.time() - tik))
    #dfTripsFAMD_DBSCAN.to_csv('./tmpOutputs/dbscan/dfTrips_famd_dbscan.csv', index=True)
    END IGNORED'''

    '''
    # gets columns related to trips made by individuals and one hot encodes categorical features
    print('\nget trip columns in hts (make dfTrips)...')
    dfTrips = htsProc.getTripsCols(dfHTS)
    print('\tnumber of columns of original trip attributes %d' % len(dfTrips.columns.tolist()))
    # sets destination, destination type and purpose of last trip to None
    # This is because almost all last trips are going home, with the exception of a few hundred trips.
    # Therefore, last destination is same as 1st origin, last destination type is home, last trip purpose is toHome.
    # htsProc.getLastTripDetails(dfTrips)

    # adds a column for the number of trips made
    print('\nadd and remove columns in dfTrips...')
    htsProc.getNumOfTripsMade(dfTrips)
    # adds columns to indicate intra-district trips and removes origin attributes of trips
    htsProc.addIntraDistTripCol(dfTrips)
    # removes columns corresponding to the origin of all trips,
    htsProc.removeOriginCols(dfTrips, origColTypes=['origTrip', 'typeOrigTrip'])
    # removes columns of destination type
    htsProc.removeDestinationCols(dfTrips, destColTypes=['destTrip', 'typeDestTrip'])
    dfTrips.to_csv('./tmpOutputs/dfTrips.csv', index=True)
    dfTrips.to_pickle('./tmpOutputs/dfTrips.pkl')
    print('\tnumber of columns after preprocessing trip attributes %d' % len(dfTrips.columns.tolist()))
    print('\tsaved files ./tmpOutputs/dfTrips.csv and ./tmpOutputs/dfTrips.pkl')

    # one hot encodes categorical trip attributes
    # notice that attributes 'origTrip', 'typeOrigTrip', 'destTrip', 'typeDestTrip' no longer exist.
    print('\noneHotEncodeTripAttribs dfTrips...')
    dfTrips = htsProc.oneHotEncodeTripAttribs(dfTrips, catAttribs=['purposeTrip', 'modeTrip'])
    dfTrips.to_csv('./tmpOutputs/dfTrips_ohe.csv', index=True)
    dfTrips.to_pickle('./tmpOutputs/dfTrips_ohe.pkl')
    print('\tnumber of columns after OHE %d' % len(dfTrips.columns.tolist()))
    print('\tsaved files ./tmpOutputs/dfTrips_ohe.csv and ./tmpOutputs/dfTrips_ohe.pkl')

    # standardises ALL columns in OHEed dfTrips
    print('\nstandardise dfTrips...')
    dfTrips = htsProc.standardise(dfTrips)
    # dfTrips['indivID'] = dfHTS['hhID'].astype(str) + '_' + dfHTS['indivID'].astype(str)
    dfTrips.to_csv('./tmpOutputs/dfTrips_ohe_std.csv', index=True)
    dfTrips.to_pickle('./tmpOutputs/dfTrips_ohe_std.pkl')
    print('\tsaved files ./tmpOutputs/dfTrips_ohe_std.csv and ./tmpOutputs/dfTrips_ohe_std.pkl')
    '''
    '''
    # finds clusters in dfTrips.
    # notes that agglomerative does not work with large datasets.
    dfTrips = pd.read_pickle('./tmpOutputs/dfTrips_ohe_std.pkl')
    print('\nrun Bisecting K-Means clustering...')
    clusteringParams = {'n_clusters': np.arange(2, 101, 1),  # np.arange(2, 101, 2),  # np.arange(2, 51, 1),
                        'init': ['k-means++'],  # ['random', 'k-means++'],
                        'n_init': [19],  # [1, 19],  # np.arange(1, 20, 2),  # np.arange(1, 6, 1),
                        'bisecting_strategy': ['biggest_inertia']  # ['largest_cluster', 'biggest_inertia']
                        }
    randStates = [711]  # 2212, 409, 311, 1103, 2611, 2910, 2911, 111, 711
    for randState in randStates:
        kwargs = {'random_state': randState}
        dfClusteringResults = clustering.findClustersBisectingKmeans(dfTrips, clusteringParams, kwargs)
        dfClusteringResults.to_pickle('./tmpOutputs/BisectingKmeans/dfClusteringResults_rand%d.pkl' % randState)
    

    # postprocesses clustering results, including building a classifier of cluster labels and feautre permutation
    randStates = [711]  # [2212, 409, 311, 1103, 2611, 2910, 2911, 111, 711]
    for randState in randStates:
        runId = 'rand%d' % randState
        print('\npostprocess clustering results runId %s...' % runId)
        dfTrips = pd.read_pickle('./tmpOutputs/dfTrips_ohe_std.pkl')
        dfClustering1Run = pd.read_pickle('./tmpOutputs/BisectingKmeans/dfClusteringResults_%s.pkl' % runId)
        # uses only the resutls associated with the maximum n_init
        dfClustering1Run = dfClustering1Run.loc[dfClustering1Run['n_init'] == dfClustering1Run['n_init'].max()]
        #clustering.postprocBisectingKmeans(dfTrips, dfClustering1Run) # NOT USED
        dfClustersAttribs = clusteringEval.calcSilhouetteScoreSamples(dfClustering1Run, dfTrips)
        dfClustering1Run = pd.merge(dfClustering1Run, dfClustersAttribs, left_on='n_clusters', right_on='n_clusters')
        dfClustering1Run.to_pickle('../myPy/tmpOutputs/BisectingKmeans/dfClusteringResults_%s_v2.pkl' % runId)
    '''

    # visualises clustering evaluation
    randStates = [711]  # [2212, 409, 311, 1103, 2611, 2910, 2911, 111, 711]
    dfTrips = pd.read_pickle('./tmpOutputs/dfTrips.pkl')
    for randState in randStates:
        runId = 'rand%d' % randState
        print(runId)
        dfClustering1Run = pd.read_pickle('./tmpOutputs/BisectingKmeans/dfClusteringResults_%s_v2.pkl' % runId)
        #dfParetoPnts = clusteringEval.plotClusteringEval1Run(dfClustering1Run,
        #                                                     '../myPy/plots/clusteringEvalv2_%s.html' % runId, runId)
        # calculates feature importance using sklearn RF => identifies variables most significant to clustering results
        clusterClassifier.calcFeatureImportancesDRF(dfTrips, dfClustering1Run, runId)
        # calculates feature importance using sklearn RF => identifies variables most significant to clustering results
        #gridSearchOutcomes = clusterClassifier.calcFeatureImportance(dfTrips, dfParetoPnts)
        #with open('gridSearchOutcomes.pkl', 'wb') as f:
        #    pickle.dump(gridSearchOutcomes, f)

    # ---------------------------------------------------------------------------------------------------------------
    # TESTING ----------------------------------------------------------------------------------------------------------
    '''
    # testing gower distance calculation
    #print(dissimilarity.calcGowerDist1Pair(dfTrips.iloc[0].to_numpy(), dfTrips.iloc[1].to_numpy()))
    #print(dissimilarity.calcGowerDist(dfTrips.iloc[[0, 1]]))
    # applies OPTICS clustering (with Gower's distance) to dfTrips. WARNINGS: OPTICS not working, out of memory problem!
    clustering.findClusters_OPTICS(dfTrips) 
    '''

    '''
    # calculates Gower's distance for all pairs in dfTrips. WARNINGS: out of memory and crashed!
    tik = time.time()
    gowerDistMat = clustering.calcGowerDist(dfTrips)
    print(type(gowerDistMat))
    print('completed calcGowerDist in %.2f seconds' % (time.time()-tik))
    '''

    '''
    n = 30000
    tik = time.time()
    randMat = np.random.random((n,n))
    print('completed generating randMat[%dx%d] in %.2f secs' % (len(randMat), len(randMat[0]), time.time()-tik))

    tik = time.time()
    with open('./tmpOutputs/randMat.pkl', 'wb') as f:
        pickle.dump(randMat, f)
    print('completed pickling randMat in %.2f secs' % (time.time() - tik))

    tik = time.time()
    with open('./tmpOutputs/randMat.pkl', 'rb') as f:
        randMat = pickle.load(f)
    print('completed loading randMat[%dx%d] in %.2f secs' % (len(randMat), len(randMat[0]), time.time() - tik))
    '''

    print('yay')


# ======================================================================================================================
def oldMain():
    dfCensusHhold = censusProc.readCensusData(censusProc.censusHholdV_sav, censusProc.censusHholdV_pkl)
    print('dfCensusHhold')
    print(dfCensusHhold.head(10))

    '''
    censusEngCSV = '../census/100421_Census2019_householdRecordsEng.csv'
    censusEngPKL = '../census/100421_Census2019_householdRecordsEng.pkl'
    if os.path.exists(censusEngPKL):
        dfCensus = pickle.load(censusEngPKL)
    else:
        dfCensus = pd.read_csv(censusEngCSV)
        dfCensus.to_pickle(censusEngPKL)
    '''

    '''
    dfCensusInd, metadata = censusProc.readIndivCensusData(censusProc.censusIndV_sav)
    dfCensusHCMC = dfCensusInd.loc[dfCensusInd['Province']==79]
    dfCensusHCMC.to_csv('dfCensusHCMC', index=False)
    '''
