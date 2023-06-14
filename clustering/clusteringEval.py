import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_samples

from bokeh.plotting import figure, output_file, show, save
from bokeh.models import Legend
from bokeh.layouts import gridplot
import bokeh.palettes
import matplotlib.pyplot as plt

# ======================================================================================================================
def calcSilhouetteScoreSamples(dfClustering1Run, dfTrips):
    def findIndices(lst, item):
        return [tmpi for tmpi, x in enumerate(lst) if x == item]

    dfClustersAttribs = pd.DataFrame()
    for index, row in dfClustering1Run.iterrows():
        nClusters = row['n_clusters']
        avgSilScore = row['silhouetteScore']
        clusterLabels = row['clusterLabels']
        uClusterLabels = set(clusterLabels)
        sampleSilhouetteVals = silhouette_samples(dfTrips, clusterLabels)
        print('calcSilhouetteScoreSamples nClusters %d, avgSilhouetteScore %.6f' % (nClusters, avgSilScore))

        clSizes = []
        clMedSilScores = []
        for lbl in uClusterLabels:  # for each cluster label in this clutering
            # Aggregate the silhouette scores for samples belonging to this cluster, and sort them
            indicesThisLabel = findIndices(clusterLabels, lbl)
            ithClusterSilhouetteVals = np.array([sampleSilhouetteVals[index] for index in indicesThisLabel])
            ithClusterSilhouetteVals.sort()
            size_cluster_i = ithClusterSilhouetteVals.shape[0]
            clSizes.append(size_cluster_i)
            clMedSilScores.append(np.median(ithClusterSilhouetteVals))
            print('\t cluster %d, size %d, mean %.6f, median %.6f' %
                  (lbl, size_cluster_i, np.mean(ithClusterSilhouetteVals), np.median(ithClusterSilhouetteVals)))

        dfClustersAttribs = pd.concat([dfClustersAttribs,
                                       pd.DataFrame.from_dict({'n_clusters': [nClusters],
                                                               'clSizes': [clSizes],
                                                               'clMedSilScores': [clMedSilScores],
                                                               'sampleSilhouette': [sampleSilhouetteVals]})])

    return dfClustersAttribs

# ======================================================================================================================
def plotSilhouetteScoreSamples(dfClustering1Run, runId):
    def findIndices(lst, item):
        return [tmpi for tmpi, x in enumerate(lst) if x == item]

    # dfClustering1Run includes clustering results for 1 random state and for 1 value of n_init (which is 19).
    for index, row in dfClustering1Run.iterrows():
        nClusters = row['n_clusters']
        avgSilScore = row['silhouetteScore']
        clusterLabels = row['clusterLabels']
        uClusterLabels = set(clusterLabels)
        sampleSilhouetteVals = row['sampleSilhouette']
        print('plotSilhouetteScoreSamples nClusters %d, avgSilhouetteScore %.6f' % (nClusters, avgSilScore))

        # initiates matplotlib axes for plotting
        #ax = plt.axes()
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.set_xlim([-0.1, 1])
        #ax.set_ylim([0, dfTrips.shape[0] + (nClusters + 1) * 10])
        yLower = 10
        for lbl in uClusterLabels:  # for each cluster label in this clutering
            # Aggregate the silhouette scores for samples belonging to this cluster, and sort them
            indicesThisLabel = findIndices(clusterLabels, lbl)
            ithClusterSilhouetteVals = np.array([sampleSilhouetteVals[index] for index in indicesThisLabel])
            ithClusterSilhouetteVals.sort()
            size_cluster_i = ithClusterSilhouetteVals.shape[0]
            print('\t cluster %d, size %d, mean %.6f, median %.6f' %
                  (lbl, size_cluster_i, np.mean(ithClusterSilhouetteVals), np.median(ithClusterSilhouetteVals)))

            yUpper = yLower + size_cluster_i
            # starts plotting
            ax.fill_betweenx(np.arange(yLower, yUpper), 0, ithClusterSilhouetteVals, alpha=.7)
            # Label the silhouette plots with their cluster numbers at the middle
            #ax.text(-0.05, yLower + 0.5 * size_cluster_i, str(lbl))
            # Compute the new y_lower for next plot
            yLower = yUpper + 10  # 10 for the 0 samples

        ax.set_title("The silhouette plot for the various clusters.")
        ax.set_xlabel("The silhouette coefficient values")
        ax.set_ylabel("Cluster label")
        # The vertical line for average silhouette score of all the values
        ax.axvline(x=avgSilScore, color="red", linestyle="--")
        ax.set_yticks([])  # Clear the yaxis labels / ticks
        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.savefig('../myPy/plots/silScoreSamples_%s/%d_Clusters.png' % (runId, nClusters))


# ======================================================================================================================
def plotClusteringEval1Run(dfClustering1Run, outFilename, runId):
    plot_width = 800
    plot_height = 400
    lineWidth = 1.5
    cSize = 7
    alpha = .7
    wBox = .25

    def mkSilhouetteScoresPlot():
        p = figure(plot_width=plot_width, plot_height=plot_height, title='Silhouette scores', toolbar_location='below')
        legendList = []

        # Plot sample silhouette scores
        # plot the whiskers of sample sihouette scores
        uSegmentBox = p.segment(dfClustering1Run['n_clusters'], dfClustering1Run['upMedSilScores'],
                                dfClustering1Run['n_clusters'], dfClustering1Run['q3MedSilScores'], line_color='black')
        lSegmentBox = p.segment(dfClustering1Run['n_clusters'], dfClustering1Run['loMedSilScores'],
                                dfClustering1Run['n_clusters'], dfClustering1Run['q1MedSilScores'], line_color='black')
        # plot the boxes of sample silhouette scores
        uBarBox = p.vbar(dfClustering1Run['n_clusters'], wBox,
                         dfClustering1Run['q2MedSilScores'], dfClustering1Run['q3MedSilScores'],
                         fill_color='white', line_color='black')
        lBarBox = p.vbar(dfClustering1Run['n_clusters'], wBox,
                         dfClustering1Run['q1MedSilScores'], dfClustering1Run['q2MedSilScores'],
                         fill_color='white', line_color='black')
        legendList.append(('median sample Silhouette scores', [uSegmentBox, lSegmentBox, uBarBox, lBarBox]))

        # Plot average Silhouette scores
        line = p.line(dfClustering1Run['n_clusters'], dfClustering1Run['silhouetteScore'],
                      line_width=lineWidth, color='blue', alpha=alpha)
        dots = p.circle(dfClustering1Run['n_clusters'], dfClustering1Run['silhouetteScore'],
                        size=cSize, color='blue', alpha=alpha)
        legendList.append(('average Silhouette scores', [line, dots]))

        p.xaxis.axis_label = 'Number of clusters'
        p.yaxis.axis_label = 'Silhouette score'
        #legend = Legend(items=legendList)
        #legend.click_policy = 'hide'
        #p.add_layout(legend, 'right')
        return p

    def plotClusterSizes():
        p = figure(plot_width=plot_width, plot_height=plot_height, title='Cluster sizes', toolbar_location='below')
        # plot the whiskers
        p.segment(dfClustering1Run['n_clusters'], dfClustering1Run['upClSizes'],
                  dfClustering1Run['n_clusters'], dfClustering1Run['q3ClSizes'], line_color='black')
        p.segment(dfClustering1Run['n_clusters'], dfClustering1Run['loClSizes'],
                  dfClustering1Run['n_clusters'], dfClustering1Run['q1ClSizes'], line_color='black')
        # plot the boxes
        p.vbar(dfClustering1Run['n_clusters'], wBox, dfClustering1Run['q2ClSizes'], dfClustering1Run['q3ClSizes'],
               fill_color='white', line_color='black')
        p.vbar(dfClustering1Run['n_clusters'], wBox, dfClustering1Run['q1ClSizes'], dfClustering1Run['q2ClSizes'],
               fill_color='white', line_color='black')
        p.xaxis.axis_label = 'Number of clusters'
        p.yaxis.axis_label = 'Cluster size'
        return p

    def plotCalinskiHarabaszScores():
        p = figure(plot_width=plot_width, plot_height=plot_height, title='Calinski-Harabasz score',
                   toolbar_location='below')
        p.line(dfClustering1Run['n_clusters'], dfClustering1Run['chScore'],
               line_width=lineWidth, color='blue', alpha=alpha)
        p.circle(dfClustering1Run['n_clusters'], dfClustering1Run['chScore'], size=cSize, color='blue', alpha=alpha)
        p.xaxis.axis_label = 'Number of clusters'
        p.yaxis.axis_label = 'Calinski-Harabasz score'
        return p

    def plotDavisBouldinScores():
        p = figure(plot_width=plot_width, plot_height=plot_height, title='Davies-Bouldin score',
                   toolbar_location='below')
        p.line(dfClustering1Run['n_clusters'], dfClustering1Run['dbScore'],
               line_width=lineWidth, color='blue', alpha=alpha)
        p.circle(dfClustering1Run['n_clusters'], dfClustering1Run['dbScore'], size=cSize, color='blue', alpha=alpha)
        p.xaxis.axis_label = 'Number of clusters'
        p.yaxis.axis_label = 'Davies-Bouldin score'
        return p

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

    def plotClSizesVsMedSilScores():
        p = figure(plot_width=plot_width, plot_height=plot_height, title='IQR cluster sizes vs IQR Silhouette scores',
                   toolbar_location='below')
        p.circle(dfClustering1Run['iqrMedSilScores'], dfClustering1Run['iqrClSizes'],
                 size=cSize, color='blue', alpha=alpha)

        colSilScores = 'stdMedSilScores'  # 'iqrMedSilScores'
        colClSizes = 'stdClSizes'  # 'iqrClSizes'
        dfIQRs = dfClustering1Run[['n_clusters', colSilScores, colClSizes,
                                   'silhouetteScore', 'clusterLabels', 'sampleSilhouette']]
        arrPoints = dfIQRs[[colSilScores, colClSizes]].to_numpy()
        dfIQRs['isPareto'] = is_pareto_efficient(arrPoints)
        dfPareto = dfIQRs.loc[dfIQRs['isPareto']]
        dfPareto.sort_values(by=[colSilScores], inplace=True)
        print('\nPareto points')
        print(dfPareto[['n_clusters', colSilScores, colClSizes, 'isPareto']])

        p.circle(dfPareto[colSilScores], dfPareto[colClSizes], size=cSize, color='red', alpha=alpha)
        p.line(dfPareto[colSilScores], dfPareto[colClSizes], line_width=lineWidth, color='red', alpha=alpha)

        p.xaxis.axis_label = 'silhouette scores variability'  # 'IQR median silhouette scores'
        p.yaxis.axis_label = 'cluster sizes variability'  # 'IQR cluster sizes'
        return p, dfPareto

    dfClustering1Run['q1MedSilScores'] = dfClustering1Run['clMedSilScores'].apply(lambda x: np.quantile(x, .25))
    dfClustering1Run['q2MedSilScores'] = dfClustering1Run['clMedSilScores'].apply(lambda x: np.quantile(x, .5))
    dfClustering1Run['q3MedSilScores'] = dfClustering1Run['clMedSilScores'].apply(lambda x: np.quantile(x, .75))
    dfClustering1Run['stdMedSilScores'] = dfClustering1Run['clMedSilScores'].apply(lambda x: np.std(x))
    dfClustering1Run['iqrMedSilScores'] = dfClustering1Run['q3MedSilScores'] - dfClustering1Run['q1MedSilScores']
    dfClustering1Run['upMedSilScores'] = dfClustering1Run.apply(
        lambda row: min(row['q3MedSilScores'] + 1.5 * row['iqrMedSilScores'], max(row['clMedSilScores'])), axis=1)
    dfClustering1Run['loMedSilScores'] = dfClustering1Run.apply(
        lambda row: max(row['q1MedSilScores'] - 1.5 * row['iqrMedSilScores'], min(row['clMedSilScores'])), axis=1)

    dfClustering1Run['q1ClSizes'] = dfClustering1Run['clSizes'].apply(lambda x: np.quantile(x, .25))
    dfClustering1Run['q2ClSizes'] = dfClustering1Run['clSizes'].apply(lambda x: np.quantile(x, .5))
    dfClustering1Run['q3ClSizes'] = dfClustering1Run['clSizes'].apply(lambda x: np.quantile(x, .75))
    dfClustering1Run['stdClSizes'] = dfClustering1Run['clSizes'].apply(lambda x: np.std(x))
    dfClustering1Run['iqrClSizes'] = dfClustering1Run['q3ClSizes'] - dfClustering1Run['q1ClSizes']
    dfClustering1Run['upClSizes'] = dfClustering1Run.apply(
        lambda row: min(row['q3ClSizes'] + 1.5 * row['iqrClSizes'], max(row['clSizes'])), axis=1)
    dfClustering1Run['loClSizes'] = dfClustering1Run.apply(
        lambda row: max(row['q1ClSizes'] - 1.5 * row['iqrClSizes'], min(row['clSizes'])), axis=1)

    dfClustering1Run.sort_values(by=['n_clusters'], inplace=True)
    pSilhouette = mkSilhouetteScoresPlot()
    pClSizes = plotClusterSizes()
    pCH = plotCalinskiHarabaszScores()
    pBD = plotDavisBouldinScores()
    pIQRs, dfParetoPnts = plotClSizesVsMedSilScores()
    #grid = gridplot([[pSilhouette, pCH], [pClSizes, pBD], [pIQRs, None]])
    grid = gridplot([[pSilhouette],
                     [pClSizes],
                     [pIQRs],
                     [pCH],
                     [pBD]])
    plotSilhouetteScoreSamples(dfParetoPnts, runId)
    output_file(outFilename, title='clustering evaluation Bisecting K-means')
    show(grid)
    return dfParetoPnts

# ======================================================================================================================
def plotClusteringEvalScoresAllRuns(dfClusteringAllRuns, outFilename):
    def mkBoxPlots(df, score2Plot, score_desc):
        """
        source: https://docs.bokeh.org/en/latest/docs/gallery/boxplot.html
        outliers not plotted in this function.
        :param score_desc:
        :param df:
        :param score2Plot:
        :return:
        """
        dfq1 = df[['n_clusters', score2Plot]].groupby(by=['n_clusters']).quantile(q=.25)
        dfq2 = df[['n_clusters', score2Plot]].groupby(by=['n_clusters']).quantile(q=.5)
        dfq3 = df[['n_clusters', score2Plot]].groupby(by=['n_clusters']).quantile(q=.75)
        dfmin = df[['n_clusters', score2Plot]].groupby(by=['n_clusters']).min()
        dfmax = df[['n_clusters', score2Plot]].groupby(by=['n_clusters']).max()

        dfStats = pd.DataFrame({'q1': dfq1[score2Plot], 'q2': dfq2[score2Plot], 'q3': dfq3[score2Plot],
                                'min': dfmin[score2Plot], 'max': dfmax[score2Plot]}, index=dfq1.index)
        dfStats['upper'] = dfStats.apply(lambda row: row['q3'] + 1.5 * (row['q3'] - row['q1']), axis=1)
        dfStats['lower'] = dfStats.apply(lambda row: row['q1'] - 1.5 * (row['q3'] - row['q1']), axis=1)
        dfStats.reset_index(inplace=True, drop=False)

        p = figure(plot_width=plotWidth, plot_height=plotHeight,  # x_axis_type="datetime",
                   title=score_desc, toolbar_location='below')
        # p = figure(tools="", background_fill_color="white")
        # plot the whiskers
        p.segment(dfStats['n_clusters'], dfStats['upper'], dfStats['n_clusters'], dfStats['q3'], line_color='black')
        # p.rect(dfStats['n_clusters'], dfStats['upper'], .2, .01, line_color='black')
        p.segment(dfStats['n_clusters'], dfStats['lower'], dfStats['n_clusters'], dfStats['q1'], line_color='black')
        # p.rect(dfStats['n_clusters'], dfStats['lower'], .2, .01, line_color='black')
        # plot the boxes
        p.vbar(dfStats['n_clusters'], 1.5, dfStats['q2'], dfStats['q3'], fill_color='white', line_color='black')
        p.vbar(dfStats['n_clusters'], 1.5, dfStats['q1'], dfStats['q2'], fill_color='white', line_color='black')
        #show(p)
        p.xaxis.axis_label = 'Number of clusters'
        p.yaxis.axis_label = score_desc
        return p

    plotWidth = 800
    plotHeight = 300
    plotArray = []
    for score, scoreDesc in zip(['silhouetteScore', 'chScore', 'dbScore'],
                                ['Silhouette score', 'Calinski-Harabasz score', 'Davies-Bouldin score']):
        plot = mkBoxPlots(dfClusteringAllRuns, score, scoreDesc)
        plotArray.append(plot)
    grid = gridplot([[plotArray[0]],
                     [plotArray[1]],
                     [plotArray[2]]])
    output_file(outFilename, title='clustering evaluation Bisecting K-means')
    show(grid)


# ======================================================================================================================
def plotClusteringEvalScoresEachRun(dfClusteringAllRuns, outFilename):
    plot_width = 800
    plot_height = 300
    lineWidth = 1.5
    cSize = 5
    alpha = .9

    def mkBokehPlot(dfResults, plotTitle, score2Plot, yAxisLabel):
        p = figure(plot_width=plot_width, plot_height=plot_height,  # x_axis_type="datetime",
                   title=plotTitle, toolbar_location='below')
        legendList = []
        mypalettes = bokeh.palettes.d3['Category20'][20][:len(dfResults['run'].unique())]
        for run, colour in zip(dfResults['run'].unique(), mypalettes):
            dfTmp = dfResults.loc[dfResults['run'] == run]
            dfTmp.sort_values(by=['n_clusters'], inplace=True)
            line = p.line(dfTmp['n_clusters'], dfTmp[score2Plot], line_width=lineWidth, color=colour, alpha=alpha)
            dots = p.circle(dfTmp['n_clusters'], dfTmp[score2Plot], size=cSize, color=colour, alpha=alpha)
            legendList.append(('%s' % run, [line, dots]))
        p.xaxis.axis_label = 'Number of clusters'
        p.yaxis.axis_label = yAxisLabel
        legend = Legend(items=legendList)
        legend.click_policy = 'hide'
        p.add_layout(legend, 'right')
        return p

    plotArray = []
    for score, scoreDesc in zip(['silhouetteScore', 'chScore', 'dbScore'],
                                ['Silhouette score', 'Calinski-Harabasz score', 'Davies-Bouldin score']):
        plot = mkBokehPlot(dfClusteringAllRuns, scoreDesc, score, scoreDesc)
        plotArray.append(plot)
    grid = gridplot([[plotArray[0]], [plotArray[1]], [plotArray[2]]])
    output_file(outFilename, title='clustering evaluation Bisecting K-means')
    show(grid)


# ======================================================================================================================
def plotClusteringEvalScores_v2(dfClustering1Run, outFilename):
    plot_width = 800
    plot_height = 300
    lineWidth = 1.5
    cSize = 7
    alpha = .9

    def mkBokehPlot(dfResults, plotTitle, score2Plot, yAxisLabel):
        p = figure(plot_width=plot_width, plot_height=plot_height,  # x_axis_type="datetime",
                   title=plotTitle, toolbar_location='below')
        legendList = []
        mypalettes = bokeh.palettes.d3['Category20'][20][:len(dfResults['n_init'].unique())]
        for n_initVal, colour in zip(dfResults['n_init'].unique(), mypalettes):
            dfTmp = dfResults.loc[dfResults['n_init'] == n_initVal]
            dfTmp.sort_values(by=['n_clusters'], inplace=True)
            line = p.line(dfTmp['n_clusters'], dfTmp[score2Plot], line_width=lineWidth, color=colour, alpha=alpha)
            dots = p.circle(dfTmp['n_clusters'], dfTmp[score2Plot], size=cSize, color=colour, alpha=alpha)
            legendList.append(('n_init %d' % n_initVal, [line, dots]))
        p.xaxis.axis_label = 'Number of clusters'
        p.yaxis.axis_label = yAxisLabel
        legend = Legend(items=legendList)
        legend.click_policy = 'hide'
        p.add_layout(legend, 'right')
        return p

    plotArray = []
    for score, scoreDesc in zip(['silhouetteScore', 'chScore', 'dbScore'],
                                ['Silhouette score', 'Calinski-Harabasz score', 'Davies-Bouldin score']):
        plot = mkBokehPlot(dfClustering1Run, scoreDesc, score, scoreDesc)
        plotArray.append(plot)
    grid = gridplot([[plotArray[0]], [plotArray[1]], [plotArray[2]]])
    output_file(outFilename, title='clustering evaluation Bisecting K-means')
    show(grid)
