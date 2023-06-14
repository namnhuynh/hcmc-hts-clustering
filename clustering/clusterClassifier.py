import pandas as pd
import numpy as np
import random
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, log_loss

import h2o
from h2o.estimators import H2ORandomForestEstimator

from bokeh.plotting import figure, output_file, show, save
from bokeh.models import Legend
from bokeh.layouts import gridplot
import bokeh.palettes

def buildDRFClassifier(dfTripsLbled, predictorCols, responseCol):
    h2o.init()
    h2o.no_progress()

    h2odfTripsLbled = h2o.H2OFrame(dfTripsLbled)
    drf = H2ORandomForestEstimator(nfolds=5, ntrees=500, max_depth=5, keep_cross_validation_predictions=True, seed=0,
                                   )  # calibrate_model=True, calibration_frame=dfTest)
    drf.train(predictorCols, responseCol, training_frame=h2odfTripsLbled)

    #print('\nmodel_performance')
    #drfPerf = drf.model_performance()

    #print('\ncross_validation_predictions')
    drfCVPredictions = drf.cross_validation_predictions()
    cvMacroAUCs = []
    for predThisFold in drfCVPredictions:
        dfPrediction = predThisFold.as_data_frame()
        clCols = [col for col in dfPrediction.columns if 'cl' in col]
        dfPrediction['sumProb'] = dfPrediction[clCols].sum(axis=1)  # calculate total predicted probability
        dfTripsLbledPred = dfTripsLbled[responseCol].to_frame()
        for col in dfPrediction.columns:
            dfTripsLbledPred[col] = dfPrediction[col]
        # test set in this fold is comprised of rows that have non-zero total predicted probability
        selCols = clCols + ['sumProb', responseCol]
        dfTestPredProb = dfTripsLbledPred[selCols].loc[dfTripsLbledPred['sumProb'] > .1]
        # removes prefix 'cl' in values in responseCol in order to use sklearn.roc_auc_score
        dfTestPredProb[responseCol] = dfTestPredProb[responseCol].apply(lambda x: int(x.replace('cl', '')))
        print(('dfTestPredProb[responseCol].unique()', dfTestPredProb[responseCol].unique()))
        dfTmp = dfTestPredProb[clCols]
        print(('dfTmp.columns', dfTmp.columns))
        macroAUC = roc_auc_score(y_true=dfTestPredProb[responseCol].to_ndfTumpy(),
                                 y_score=dfTestPredProb[clCols].to_numpy(),
                                 average='macro', multi_class='ovr')
        print('nTestRows %d (%.3f), macroAUC %.3f' %
              (dfTestPredProb.shape[0], dfTestPredProb.shape[0]/dfPrediction.shape[0], macroAUC))
        cvMacroAUCs.append(macroAUC)

    return cvMacroAUCs


def buildDRFClassifier_v2(dfTripsLbled, predictorCols, responseCol):
    h2o.init()
    h2o.no_progress()

    nRandomRuns = 5
    cvMacroAUCs = []
    loglossList = []
    for iRun in range(nRandomRuns):
        dfTrain, dfTest = mkTrainTestSets(dfTripsLbled, responseCol, .2)
        hfTrain = h2o.H2OFrame(dfTrain)
        hfTest = h2o.H2OFrame(dfTest)
        drf = H2ORandomForestEstimator(ntrees=500, max_depth=5, seed=0)
        drf.train(x=predictorCols, y=responseCol, training_frame=hfTrain, validation_frame=hfTest)
        #varImp = drf.varimp(use_pandas=True)
        # gets prediction on dfTest and convert it to dataframe
        dfPrediction = drf.predict(hfTest).as_data_frame()
        dfPrediction.index = dfTest.index  # this is needed to correcly assign dfPrediction columns to dfTestPredProb
        # gets list of columns of predicted probability of each label
        # (columns in dfPrediction are 'predict', 'cl0', 'cl1', ...)
        # must do it here because 'cl' also exists in responseCol ('clusterLabels')
        clCols = [col for col in dfPrediction.columns if 'cl' in col]
        # merges responseCol in dfTest with columns in dfPrediction
        dfTestPredProb = dfTest[responseCol].to_frame()
        for col in dfPrediction.columns:
            dfTestPredProb[col] = dfPrediction[col]
        # removes prefix 'cl' in values in responseCol in order to use sklearn.roc_auc_score
        dfTestPredProb[responseCol] = dfTestPredProb[responseCol].apply(lambda x: int(x.replace('cl', '')))
        if len(clCols) == 2:
            macroAUC = roc_auc_score(y_true=dfTestPredProb[responseCol].to_numpy(),
                                     y_score=dfTestPredProb['cl1'].to_numpy(),
                                     average='macro', multi_class='ovr')
        else:
            macroAUC = roc_auc_score(y_true=dfTestPredProb[responseCol].to_numpy(),
                                     y_score=dfTestPredProb[clCols].to_numpy(),
                                     average='macro', multi_class='ovr')
        logloss = log_loss(y_true=dfTestPredProb[responseCol].to_numpy(),
                           y_pred=dfTestPredProb[clCols].to_numpy())
        print('\t\tiRun %d, nTestRows %d (%.3f), macroAUC %.6f, logloss %.6f' %
              (iRun, dfTestPredProb.shape[0], dfTestPredProb.shape[0] / dfTripsLbled.shape[0], macroAUC, logloss))
        cvMacroAUCs.append(macroAUC)
        loglossList.append(logloss)

    drf = H2ORandomForestEstimator(ntrees=500, max_depth=5, seed=0)
    drf.train(x=predictorCols, y=responseCol, training_frame=h2o.H2OFrame(dfTripsLbled))
    varImp = drf.varimp(use_pandas=True)

    return cvMacroAUCs, loglossList, varImp


def calcFeatureImportancesDRF(dfTrips, dfClustering1Run, runId):
    def is_pareto_efficient(costs, return_mask=True):
        """
        Find the pareto-efficient points.
        Source: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
        :param costs: An (n_points, n_costs) array
        :param return_mask: True to return a mask
        :return: An array of indices of pareto-efficient points.
            If return_mask is True, this will be an (n_points, ) boolean array
            Otherwise it will be a (n_efficient_points, ) integer array of indices.
        """
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index < len(costs):
            nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype=bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient

    dfClustering1Run['stdMedSilScores'] = dfClustering1Run['clMedSilScores'].apply(lambda x: np.std(x))
    dfClustering1Run['stdClSizes'] = dfClustering1Run['clSizes'].apply(lambda x: np.std(x))
    arrPoints = dfClustering1Run[['stdMedSilScores', 'stdClSizes']].to_numpy()
    dfClustering1Run['isPareto'] = is_pareto_efficient(arrPoints)
    dfPareto = dfClustering1Run.loc[dfClustering1Run['isPareto']]
    dfPareto.sort_values(by=['n_clusters'], inplace=True)

    responseCol = 'clusterLabels'
    featureCols = dfTrips.columns.tolist()
    random.seed(0)
    print(dfPareto[['n_clusters', 'stdMedSilScores', 'stdClSizes', 'isPareto']])
    for idx, row in dfPareto.iterrows():  # for each of the numbers of clusters in dfParetoPnts
        clusterLabels = row['clusterLabels']
        nClusters = row['n_clusters']
        #if nClusters < 97:
        #    continue
        print('\n\ttrain a classifier for clustering results - n_clusters %d' % nClusters)
        # assigns cluster labels to dfTrips
        dfTrips[responseCol] = clusterLabels
        dfTrips[responseCol] = dfTrips[responseCol].apply(lambda x: 'cl%d' % x)
        #print(dfTrips[['purposeTrip1', 'modeTrip1', 'nMinsTrip1', 'nMinsDepTimeTrip1', 'clusterLabels']].head())
        # trains a random forest classifier of cluster labels
        # notes that we use the whole of dfTrips for training, without splitting it into train set and test set.
        # this is because we only use the classifier to understand feature importance, not for predicting purposes.
        tik = time.time()
        #cvMacroAUCs = buildDRFClassifier(dfTrips, featureCols, responseCol)
        cvMacroAUCs, loglossList, varImp = buildDRFClassifier_v2(dfTrips, featureCols, responseCol)
        print(('cvMacroAUCs', cvMacroAUCs))
        print(('loglossList', loglossList))
        print('\tvarImp')
        print(type(varImp))
        print(varImp)
        print('\tcompleted buildDRFClassifier in %.2f secs - avgMacroAUCs %.6f, avgLogloss %.6f' %
              (time.time() - tik, np.average(np.array(cvMacroAUCs)), np.average(np.array(loglossList))))
        varImp.to_csv('./tmpOutputs/drfVarImportance/%s/%dClusters.csv' % (runId, nClusters), index=True)

# ======================================================================================================================
def mkTrainTestSets(dfTrips, responseCol, pcTestSet):
    # splits dfTrips into train set and test set
    # note that we use stratified sampling to preserve the proportion of classes (clustes) in train set and test set.
    testSet = pd.DataFrame()
    uLabels = dfTrips[responseCol].unique()  # gets the list of unique cluster labels
    # randomly draws rows from the subset of dfTrips corresponding to each label for test set
    for label in uLabels:
        dfTripsThisLbl = dfTrips.loc[dfTrips[responseCol] == label]
        # randomly draws test set from dfTripsThisLb
        shuffledIndex = random.sample(dfTripsThisLbl.index.tolist(), len(dfTripsThisLbl.index.tolist()))
        testIndices = shuffledIndex[:int(len(shuffledIndex) * pcTestSet)]
        if len(testIndices) > 0:
            dfTmpTest = dfTripsThisLbl.loc[testIndices]
            testSet = pd.concat([testSet, dfTmpTest])
        # print('total %d, test %d' % (dfTmp.shape[0], dfTmpTest.shape[0]))
    # makes train set which contains the remaining rows in dfTrips
    trainSet = dfTrips[~dfTrips.index.isin(testSet.index)]

    # makes sure that all labels in trainSet are also present in testSet
    for uLblTrain in trainSet[responseCol].unique():
        if uLblTrain not in testSet[responseCol].values.tolist():
            dfTrainSub = trainSet.loc[trainSet[responseCol] == uLblTrain]
            if dfTrainSub.shape[0] == 1:
                testSet = pd.concat([testSet, dfTrainSub])
            else:
                # randomly picks 1 row from dfTrainSub and adds it to testSet
                shuffledIdxTrainSub = random.sample(dfTrainSub.index.tolist(), len(dfTrainSub.index.tolist()))
                idx2Pick = shuffledIdxTrainSub[:1]
                testSet = pd.concat([testSet, dfTrainSub.loc[idx2Pick]])
                # remove idx2Pick from trainSet
                trainSet.drop(idx2Pick, inplace=True)

    return trainSet, testSet

# ======================================================================================================================
def trainRF(dfTrain, predictorCols, responseCol, gridParams):
    # hyperparameter tuning the random forest model through cross validation on dfTrain
    gridSearchRF = GridSearchCV(estimator=RandomForestClassifier(),
                                param_grid=gridParams,
                                scoring='roc_auc_ovo',  # 'roc_auc_ovo', 'neg_log_loss'
                                cv=10, n_jobs=-1)
    gridSearchRF.fit(X=dfTrain[predictorCols], y=dfTrain[responseCol])

    return gridSearchRF.cv_results_, gridSearchRF.best_params_

# ======================================================================================================================
def calcFeatureImportance(dfTrips, dfParetoPnts):
    gridParams = {'n_estimators': [1000],  # np.arange(100, 501, 100),
                  # 'criterion': ['gini', 'entropy', 'log_loss'],
                  # 'max_depth': np.arange(1, 11, 1),
                  # 'max_features': np.arange(2, 11, 1),  # np.arange(7, int(len(featureCols)/2)+1, 1),
                  'max_samples': [.1, .15, .2, .25, .3, .35, .4, .45, .5]}
    responseCol = 'clusterLabels'
    featureCols = dfTrips.columns.tolist()

    gridSearchOutcomes = {}
    for idx, row in dfParetoPnts.iterrows():  # for each of the numbers of clusters in dfParetoPnts
        clusterLabels = row['clusterLabels']
        nClusters = row['n_clusters']
        print('\n\ttrain a classifier for clustering results - n_clusters %d' % nClusters)
        # assigns cluster labels to dfTrips
        dfTrips[responseCol] = clusterLabels
        # trains a random forest classifier of cluster labels
        # notes that we use the whole of dfTrips for training, without splitting it into train set and test set.
        # this is because we only use the classifier to understand feature importance, not for predicting purposes.
        tik = time.time()
        cvResults, bestParams = trainRF(dfTrips, featureCols, responseCol, gridParams)
        print('\tcompleted trainRF in %.2f secs' % (time.time() - tik))
        # trains the best RF
        bestRFModel = RandomForestClassifier(n_estimators=bestParams['n_estimators'],
                                             max_samples=bestParams['max_samples']).fit(X=dfTrips[featureCols],
                                                                                         y=dfTrips[responseCol])
        # fetches the impurity-based feature importances
        featureImportances = bestRFModel.feature_importances_
        gridSearchOutcomes[nClusters] = {'cvResults': cvResults,
                                         'bestParams': bestParams,
                                         'featureImportances': featureImportances}

    return gridSearchOutcomes

# ======================================================================================================================
def plotFeatureImportances(gridSearchOutcomes):
    dfFeatureImportances = pd.DataFrame()
    p = figure(plot_width=600, plot_height=1000, title='RF classifier feature importances ', toolbar_location='below')
    legendList = []
    #p.hbar()
