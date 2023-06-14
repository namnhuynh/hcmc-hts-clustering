import bokeh.palettes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bokeh.layouts import gridplot
from bokeh.models import Label, FactorRange
from bokeh.plotting import figure, output_file, show


# ======================================================================================================================
def getGroupByVar2Plot(dfTripsLbled, clusterLbl, var2Plot, allValsVar2Plot):
    dfTripsThisCluster = dfTripsLbled.loc[dfTripsLbled['clusterLabels'] == clusterLbl]
    # fTripsLbled.groupby(by=['clusterLabels']).count()['purposeTrip1']
    dfGroupByVar2Plot = dfTripsThisCluster.groupby(by=[var2Plot]).count()['clusterLabels']
    # renames the Series to 'counts' and converts it to a dataframe
    dfGroupByVar2Plot = dfGroupByVar2Plot.rename('counts', inplace=True).to_frame()
    # converts the index, currently var2Plot, into a column of the new dataframe
    dfGroupByVar2Plot.reset_index(inplace=True, drop=False)
    # tests if var2Plot has all possible values (in allValsVar2Plot)
    uVar2Plot = dfGroupByVar2Plot[var2Plot].unique().tolist()
    missingVals = list(set(allValsVar2Plot) - set(uVar2Plot))
    # adds any missing value of var2Plot with counts 0 to dfGroupByVar2Plot
    if len(missingVals) > 0:
        dfMissing = pd.DataFrame(list(zip(missingVals, [0 for i in range(len(missingVals))])),
                                 columns=[var2Plot, 'counts'])
        dfGroupByVar2Plot = pd.concat([dfGroupByVar2Plot, dfMissing])
    # sorts by var2Plot for consistent plotting
    dfGroupByVar2Plot.sort_values(by=[var2Plot], inplace=True)
    return dfGroupByVar2Plot

def plotCatVarInBiggestClusters(dfTripsLbled, dfBiggestClusters, var2Plot, plotWidth, plotHeight, alpha):
    # fetches all possible values in var2Plot
    allValsVar2Plot = dfTripsLbled[var2Plot].unique().tolist()
    allValsVar2Plot = [x for x in allValsVar2Plot if x is not None]  # removes None value

    maxCounts = 0
    plotDataOfClusters = {}
    for clusterLbl in dfBiggestClusters['clusterLabels'].unique():
        dfGroupByVar2Plot = getGroupByVar2Plot(dfTripsLbled, clusterLbl, var2Plot, allValsVar2Plot)
        maxCounts = max(maxCounts, dfGroupByVar2Plot['counts'].max())
        plotDataOfClusters[clusterLbl] = dfGroupByVar2Plot

    # starts plotting
    pList = []
    for clusterLbl, dfGroupByVar2Plot in plotDataOfClusters.items():
        nElementsThisCluster = dfBiggestClusters.loc[dfBiggestClusters['clusterLabels'] == clusterLbl]['clSize']
        p = figure(plot_width=plotWidth, plot_height=plotHeight, toolbar_location=None,  # 'below',
                   title='%s, cluster %d (%d people)' % (var2Plot, clusterLbl, nElementsThisCluster),
                   x_range=dfGroupByVar2Plot[var2Plot].values, y_range=(0, maxCounts*1.05))
        p.vbar(x=dfGroupByVar2Plot[var2Plot].values, top=dfGroupByVar2Plot['counts'].values, width=.9, alpha=alpha)
        pList.append(p)

    return gridplot([[plot for plot in pList]], toolbar_location=None)

def plotCatVarInBiggestClusters_Hor(dfTripsLbled, dfBiggestClusters, var2Plot, plotWidths, plotHeight, alpha):
    # defines the width of the 1st plot and of the remaining plots
    width1stPlot = plotWidths[0]
    widthOtherPlots = plotWidths[1]

    # fetches all possible values in var2Plot
    allValsVar2Plot = dfTripsLbled[var2Plot].unique().tolist()
    allValsVar2Plot = [x for x in allValsVar2Plot if x is not None]  # removes None value

    maxCounts = 0
    plotDataOfClusters = {}
    for clusterLbl in dfBiggestClusters['clusterLabels'].unique():
        dfGroupByVar2Plot = getGroupByVar2Plot(dfTripsLbled, clusterLbl, var2Plot, allValsVar2Plot)
        maxCounts = max(maxCounts, dfGroupByVar2Plot['counts'].max())
        plotDataOfClusters[clusterLbl] = dfGroupByVar2Plot

    # starts plotting
    pList = []
    for clusterLbl, dfGroupByVar2Plot in plotDataOfClusters.items():
        nElementsThisCluster = dfBiggestClusters.loc[dfBiggestClusters['clusterLabels'] == clusterLbl]['clSize']
        if len(pList) == 0:
            p = figure(plot_width=width1stPlot, plot_height=plotHeight, toolbar_location=None,  # 'below',
                       title='%s, cluster %d (%d people)' % (var2Plot, clusterLbl, nElementsThisCluster),
                       y_range=dfGroupByVar2Plot[var2Plot].values, x_range=(0, maxCounts*1.05))
        else:
            p = figure(plot_width=widthOtherPlots, plot_height=plotHeight, toolbar_location=None,  # 'below',
                       title='%s, cluster %d (%d people)' % (var2Plot, clusterLbl, nElementsThisCluster),
                       y_range=dfGroupByVar2Plot[var2Plot].values, x_range=(0, maxCounts * 1.05))
        if len(pList) > 0:  # if not the first plot, don't show axis label
            p.yaxis.major_label_text_font_size = '0pt'
        p.hbar(y=dfGroupByVar2Plot[var2Plot].values, right=dfGroupByVar2Plot['counts'].values, width=.9, alpha=alpha)
        pList.append(p)

    return gridplot([[plot for plot in pList]], toolbar_location=None)

def plotNumVarInBiggestClusters(dfTripsLbled, dfBiggestClusters, var2Plot, plotWidth, plotHeight, alpha):
    # starts plotting
    pList = []
    for clusterLbl in dfBiggestClusters['clusterLabels'].unique():
        nElementsThisCluster = dfBiggestClusters.loc[dfBiggestClusters['clusterLabels'] == clusterLbl]['clSize']
        valsThisCluster = dfTripsLbled.loc[dfTripsLbled['clusterLabels'] == clusterLbl][var2Plot]
        hist, edges = np.histogram(valsThisCluster, density=True, bins=50)
        p = figure(plot_width=plotWidth, plot_height=plotHeight, toolbar_location=None,
                   title='%s, cluster %d (%d people)' % (var2Plot, clusterLbl, nElementsThisCluster))
        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
               fill_color="navy", line_color="white", alpha=alpha)
        #p.line(x, pdf, line_color="#ff8888", line_width=4, alpha=0.7)  # legend_label="PDF")
        p.y_range.start = 0
        pList.append(p)
    grid = gridplot([[plot for plot in pList]], toolbar_location=None)
    return grid


def plotBoxInBiggestClusters(dfTripsLbled, dfBiggestClusters, var2Plot, xRange, plotWidths, plotHeight):
    # defines the width of the 1st plot and of the remaining plots
    width1stPlot = plotWidths[0]
    widthOtherPlots = plotWidths[1]

    pList = []
    for clusterLbl in dfBiggestClusters['clusterLabels'].unique():
        dfTripsThisCluster = dfTripsLbled.loc[dfTripsLbled['clusterLabels'] == clusterLbl]
        q1 = dfTripsThisCluster[var2Plot].quantile(.25)
        q2 = dfTripsThisCluster[var2Plot].quantile(.5)
        q3 = dfTripsThisCluster[var2Plot].quantile(.75)
        up = min(q3 + 1.5*(q3 - q1), dfTripsThisCluster[var2Plot].max())
        lo = max(q1 - 1.5*(q3 - q1), dfTripsThisCluster[var2Plot].min())
        out = dfTripsThisCluster[var2Plot].loc[(dfTripsThisCluster[var2Plot] > up) |
                                               (dfTripsThisCluster[var2Plot] < lo)]
        nElementsThisCluster = dfBiggestClusters.loc[dfBiggestClusters['clusterLabels'] == clusterLbl]['clSize']
        if len(pList) == 0:
            p = figure(plot_width=width1stPlot, plot_height=plotHeight, toolbar_location=None,
                       y_range=[var2Plot], x_range=xRange,
                       title='%s, cluster %d (%d people)' % (var2Plot, clusterLbl, nElementsThisCluster))
        else:
            p = figure(plot_width=widthOtherPlots, plot_height=plotHeight, toolbar_location=None,
                       y_range=[var2Plot], x_range=xRange,
                       title='%s, cluster %d (%d people)' % (var2Plot, clusterLbl, nElementsThisCluster))
        # stems
        p.segment(up, [var2Plot], q3, [var2Plot], line_color='black')
        p.segment(lo, [var2Plot], q1, [var2Plot], line_color='black')
        # boxes
        p.hbar([var2Plot], .25, q2, q3, line_color='black', fill_color='white')
        p.hbar([var2Plot], .25, q1, q2, line_color='black', fill_color='white')
        # outliers
        if not out.empty:
            p.circle([var2Plot] * len(out.values), list(out.values), size=6, fill_alpha=0.6)
        # if not the first plot, don't show axis label
        if len(pList) > 0:
            p.yaxis.major_label_text_font_size = '0pt'
        pList.append(p)
    return gridplot([[plot for plot in pList]], toolbar_location=None)


def getBiggestClusterLabels(dfTripsLbled, nBiggestClusters=4):
    # gets counts of rows in any column (e.g. 'purposeTrip1') corresponding to each cluster label
    # gets the 1 first column in dfTripsLbled
    col1 = dfTripsLbled.columns.tolist()[0]
    dfGroupByClusterSize = dfTripsLbled.groupby(by=['clusterLabels']).count()[col1]  # ['purposeTrip1']
    # changes the name of the returned Series into clSize and convert the Series to dataframe
    dfGroupByClusterSize = dfGroupByClusterSize.rename('clSize', inplace=True).to_frame()
    # resets index to make the current index (clusterLabels) a column
    dfGroupByClusterSize.reset_index(inplace=True, drop=False)
    # sorts dfGroupByClusterSize by clSize in descending order
    dfGroupByClusterSize.sort_values(by=['clSize'], inplace=True, ascending=False)
    #print(dfGroupByClusterSize)
    # gets the largest clusters, i.e. the 1st few rows in dfGroupByClusterSize. Note that we use iloc, not loc.
    dfLargestClusters = dfGroupByClusterSize.iloc[:nBiggestClusters]
    return dfLargestClusters  # has 2 columns, 'clusterLabels' and 'clSize'


def plotTripVarsBiggestClusters(dfTrips, dfClustering1Run, nClusters, nBiggestClusters, vars2Plot,
                                runDesc, outFilename):
    plotWidth = 300
    plotHeight = 120
    alpha = .75

    # assigns cluster labels to dfTrips
    clusterLabels = dfClustering1Run.loc[dfClustering1Run['n_clusters'] == nClusters]['clusterLabels'].values[0]
    dfTrips['clusterLabels'] = clusterLabels
    # gets the dataframe of biggest clusters, sorted in descending order
    dfBiggestClusters = getBiggestClusterLabels(dfTrips, min(nBiggestClusters, nClusters))
    print(dfBiggestClusters)
    print('%d largest clusters account for %.1f per cent of the all (%d) observations' %
          (dfBiggestClusters.shape[0], dfBiggestClusters['clSize'].sum() / dfTrips.shape[0] * 100, dfTrips.shape[0]))
    # converts certain numerical attributes categorical
    dfTrips['nTrips'] = dfTrips['nTrips'].apply(lambda x: '%dTrips' % x)  # number of trips nTrips to categorical
    for iTrip in range(1, 8):  # whether a trip is intra-district to categorial
        colName = 'intraDistTrip%d' % iTrip
        dfTrips[colName] = dfTrips[colName].apply(lambda x: 'Yes' if x == 1 else 'No')
    # export labelled dfTrips to csv for validation purposes
    dfTrips.to_csv('../myPy/plots/tripVars_%s/dfTripsLbled_%dCls_%dBiggest.csv' %
                   (runDesc, nClusters, nBiggestClusters), index=True)

    gridList = []
    for var2Plot in vars2Plot:
        if 'nMins' in var2Plot:
            plotsThisVar = plotNumVarInBiggestClusters(dfTrips, dfBiggestClusters, var2Plot,
                                                       plotWidth, plotHeight, alpha)
        else:
            plotsThisVar = plotCatVarInBiggestClusters(dfTrips, dfBiggestClusters, var2Plot,
                                                       plotWidth, plotHeight, alpha)
        gridList.append(plotsThisVar)

    grid = gridplot([[rowPlots] for rowPlots in gridList])
    output_file(outFilename, title='%d biggest of %d clusters, %s' % (nBiggestClusters, nClusters, runDesc))
    show(grid)

# ======================================================================================================================
def plotDemoAttribsBiggestClusters(dfHTSDemo, dfClustering1Run, nClusters, nBiggestClusters, runDesc, outFilename):
    defaultPlotWidth = 300
    defaultPlotHeight = 120
    alpha = .75

    # assigns cluster labels to dfTrips
    clusterLabels = dfClustering1Run.loc[dfClustering1Run['n_clusters'] == nClusters]['clusterLabels'].values[0]
    dfHTSDemo['clusterLabels'] = clusterLabels
    # gets the dataframe of biggest clusters, sorted in descending order
    dfBiggestClusters = getBiggestClusterLabels(dfHTSDemo, min(nBiggestClusters, nClusters))
    print(dfBiggestClusters)
    print('%d largest clusters account for %.1f per cent of the all (%d) observations' %
          (dfBiggestClusters.shape[0], dfBiggestClusters['clSize'].sum()/dfHTSDemo.shape[0]*100, dfHTSDemo.shape[0]))
    # replaces nan by 'N/A' or the below plotting cannot proceed.
    for col in dfHTSDemo.columns:
        dfHTSDemo[col] = dfHTSDemo[col].fillna('N/A')

    # export labelled dfHTSDemo to csv for validation purposes
    dfHTSDemo.to_csv('../myPy/plots/demoVars_%s/dfTripsLbled_%dCls_%dBiggest.csv' %
                     (runDesc, nClusters, nBiggestClusters), index=True)

    gridList = [plotBoxInBiggestClusters(dfHTSDemo, dfBiggestClusters, 'age', (0, 100), [310, 300], defaultPlotHeight),
                plotCatVarInBiggestClusters_Hor(dfHTSDemo, dfBiggestClusters, 'gender',
                                                [330, 300], defaultPlotHeight, alpha),
                plotCatVarInBiggestClusters_Hor(dfHTSDemo, dfBiggestClusters, 'licence',
                                                [310, 300], defaultPlotHeight, alpha),
                plotCatVarInBiggestClusters_Hor(dfHTSDemo, dfBiggestClusters, 'ownMotorVehicles',
                                                [310, 300], defaultPlotHeight, alpha),
                plotCatVarInBiggestClusters_Hor(dfHTSDemo, dfBiggestClusters, 'eduLevel',
                                                [400, 300], defaultPlotHeight, alpha),
                plotCatVarInBiggestClusters_Hor(dfHTSDemo, dfBiggestClusters, 'jobType',
                                                [400, 300], defaultPlotHeight, alpha),
                plotCatVarInBiggestClusters_Hor(dfHTSDemo, dfBiggestClusters, 'employStatus',
                                                [400, 300], defaultPlotHeight, alpha),
                plotCatVarInBiggestClusters_Hor(dfHTSDemo, dfBiggestClusters, 'monthlyIncome',
                                                [400, 300], defaultPlotHeight, alpha)]

    grid = gridplot([[rowPlots] for rowPlots in gridList])
    output_file(outFilename, title='%d biggest of %d clusters, %s' % (nBiggestClusters, nClusters, runDesc))
    show(grid)

# ======================================================================================================================
def plot_CH_DB_multiRuns(dfClusteringAllRuns, outFilename):
    plot_width = 300
    plot_height = 300
    lineWidth = 1.5
    cSize = 1
    alpha = .75

    def mkBokehPlot(dfResults, plotTitle, score2Plot, yAxisLabel):
        p = figure(plot_width=plot_width, plot_height=plot_height,  # x_axis_type="datetime",
                   title=plotTitle, toolbar_location='below')
        legendList = []
        #mypalettes = bokeh.palettes.d3['Category20'][20][:len(dfResults['run'].unique())]
        mypalettes = bokeh.palettes.d3['Category10'][10][:len(dfResults['run'].unique())]
        for run, colour in zip(dfResults['run'].unique(), mypalettes):
            dfTmp = dfResults.loc[dfResults['run'] == run]
            dfTmp.sort_values(by=['n_clusters'], inplace=True)
            line = p.line(dfTmp['n_clusters'], dfTmp[score2Plot], line_width=lineWidth, color=colour, alpha=alpha)
            dots = p.circle(dfTmp['n_clusters'], dfTmp[score2Plot], size=cSize, color=colour, alpha=alpha)
            legendList.append(('%s' % run, [line, dots]))
        p.xaxis.axis_label = 'Number of clusters'
        p.yaxis.axis_label = yAxisLabel
        #legend = Legend(items=legendList)
        #legend.click_policy = 'hide'
        #p.add_layout(legend, 'right')
        return p

    plotArray = []
    for score, scoreDesc in zip(['chScore', 'dbScore'],
                                ['Calinski-Harabasz score', 'Davies-Bouldin score']):
        plot = mkBokehPlot(dfClusteringAllRuns, scoreDesc, score, scoreDesc)
        plotArray.append(plot)
    #grid = gridplot([[plotArray[0]], [plotArray[1]]])
    grid = gridplot([[plotArray[0], plotArray[1]]])
    output_file(outFilename, title='DB_CH score multi runs')
    show(grid)

# ======================================================================================================================
def plotAvgSilhouetteMultiRuns(clusteringResultsPklFiles, outFilename):
    plotWidth = 250
    plotHeight = 250
    lineWidth = 1.5
    cSize = 1
    alpha = .75
    wBox = .25

    def mkSilhouetteScoresPlot():
        p = figure(plot_width=plotWidth, plot_height=plotHeight)  # title='Silhouette scores', toolbar_location='below'
        legendList = []

        # For each number of clusters, plot the boxplot of the distribution of the median of sample silhouette scores
        # of each cluster.
        # plot the whisker
        uSegmentBox = p.segment(dfClustering1Run['n_clusters'], dfClustering1Run['upMedSilScores'],
                                dfClustering1Run['n_clusters'], dfClustering1Run['q3MedSilScores'], line_color='black')
        lSegmentBox = p.segment(dfClustering1Run['n_clusters'], dfClustering1Run['loMedSilScores'],
                                dfClustering1Run['n_clusters'], dfClustering1Run['q1MedSilScores'], line_color='black')
        # plot the boxes
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
        # adds Label, more info at https://docs.bokeh.org/en/test/docs/user_guide/annotations.html
        runDescLabel = Label(x=5, y=200, x_units='screen', y_units='screen',
                             text=runDesc, render_mode='css',
                             border_line_color='black', border_line_alpha=1.0)
        p.add_layout(runDescLabel)
        #p.xaxis.axis_label = 'Number of clusters'
        #p.yaxis.axis_label = 'Silhouette score'
        # legend = Legend(items=legendList)
        # legend.click_policy = 'hide'
        # p.add_layout(legend, 'right')
        return p

    pList = []
    for file, runDesc in clusteringResultsPklFiles.items():
        dfClustering1Run = pd.read_pickle(file)
        dfClustering1Run['q1MedSilScores'] = dfClustering1Run['clMedSilScores'].apply(lambda x: np.quantile(x, .25))
        dfClustering1Run['q2MedSilScores'] = dfClustering1Run['clMedSilScores'].apply(lambda x: np.quantile(x, .5))
        dfClustering1Run['q3MedSilScores'] = dfClustering1Run['clMedSilScores'].apply(lambda x: np.quantile(x, .75))
        dfClustering1Run['iqrMedSilScores'] = dfClustering1Run['q3MedSilScores'] - dfClustering1Run['q1MedSilScores']
        dfClustering1Run['upMedSilScores'] = dfClustering1Run.apply(
            lambda row: min(row['q3MedSilScores'] + 1.5 * row['iqrMedSilScores'], max(row['clMedSilScores'])), axis=1)
        dfClustering1Run['loMedSilScores'] = dfClustering1Run.apply(
            lambda row: max(row['q1MedSilScores'] - 1.5 * row['iqrMedSilScores'], min(row['clMedSilScores'])), axis=1)
        dfClustering1Run.sort_values(by=['n_clusters'], inplace=True)
        pList.append(mkSilhouetteScoresPlot())

    grid = gridplot([[pList[0], pList[1], pList[2]],  # run 1, run 2, run 3
                     [pList[3], pList[4], pList[5]],  # run 4, run 5, run 6
                     [pList[6], pList[7], pList[8]]   # run 7, run 8, run 9
                     ])
    output_file(outFilename, title='Silhouette score multi runs')
    show(grid)

# ======================================================================================================================
def plotClusterSizesMultiRuns(clusteringResultsPklFiles, outFilename):
    plotWidth = 250
    plotHeight = 250
    wBox = .25

    def plotClusterSizes():
        p = figure(plot_width=plotWidth, plot_height=plotHeight)  # title='Cluster sizes', toolbar_location='below')
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
        # adds Label, more info at https://docs.bokeh.org/en/test/docs/user_guide/annotations.html
        runDescLabel = Label(x=150, y=200, x_units='screen', y_units='screen',
                             text=runDesc, render_mode='css',
                             border_line_color='black', border_line_alpha=1.0)
        p.add_layout(runDescLabel)
        #p.xaxis.axis_label = 'Number of clusters'
        #p.yaxis.axis_label = 'Cluster size'
        return p

    pList = []
    for file, runDesc in clusteringResultsPklFiles.items():
        dfClustering1Run = pd.read_pickle(file)
        dfClustering1Run['q1ClSizes'] = dfClustering1Run['clSizes'].apply(lambda x: np.quantile(x, .25))
        dfClustering1Run['q2ClSizes'] = dfClustering1Run['clSizes'].apply(lambda x: np.quantile(x, .5))
        dfClustering1Run['q3ClSizes'] = dfClustering1Run['clSizes'].apply(lambda x: np.quantile(x, .75))
        dfClustering1Run['iqrClSizes'] = dfClustering1Run['q3ClSizes'] - dfClustering1Run['q1ClSizes']
        dfClustering1Run['upClSizes'] = dfClustering1Run.apply(
            lambda row: min(row['q3ClSizes'] + 1.5 * row['iqrClSizes'], max(row['clSizes'])), axis=1)
        dfClustering1Run['loClSizes'] = dfClustering1Run.apply(
            lambda row: max(row['q1ClSizes'] - 1.5 * row['iqrClSizes'], min(row['clSizes'])), axis=1)
        dfClustering1Run.sort_values(by=['n_clusters'], inplace=True)
        pList.append(plotClusterSizes())

    grid = gridplot([[pList[0], pList[1], pList[2]],  # run 1, run 2, run 3
                     [pList[3], pList[4], pList[5]],  # run 4, run 5, run 6
                     [pList[6], pList[7], pList[8]]  # run 7, run 8, run 9
                     ])
    output_file(outFilename, title='Cluster sizes multi runs')
    show(grid)

# ======================================================================================================================
def plotStdevClsizesVersusMedSilScores(clusteringResultsPklFiles, outFilename):
    plotWidth = 250
    plotHeight = 250
    lineWidth = 1.5
    cSize = 4
    alpha = .75

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
        p = figure(plot_width=plotWidth, plot_height=plotHeight)
        p.circle(dfClustering1Run['stdMedSilScores'], dfClustering1Run['stdClSizes'],
                 size=cSize, color='black', alpha=alpha)
        p.circle(dfPareto['stdMedSilScores'], dfPareto['stdClSizes'], size=cSize, color='red', alpha=alpha)
        p.line(dfPareto['stdMedSilScores'], dfPareto['stdClSizes'], line_width=lineWidth, color='red', alpha=alpha)
        # adds Label, more info at https://docs.bokeh.org/en/test/docs/user_guide/annotations.html
        runDescLabel = Label(x=150, y=200, x_units='screen', y_units='screen',
                             text=runDesc, render_mode='css',
                             border_line_color='black', border_line_alpha=1.0)
        p.add_layout(runDescLabel)
        #p.xaxis.axis_label = 'silhouette scores variability'  # 'IQR median silhouette scores'
        #p.yaxis.axis_label = 'cluster sizes variability'  # 'IQR cluster sizes'
        return p

    pList = []
    for file, runDesc in clusteringResultsPklFiles.items():
        dfClustering1Run = pd.read_pickle(file)
        dfClustering1Run['stdMedSilScores'] = dfClustering1Run['clMedSilScores'].apply(lambda x: np.std(x))
        dfClustering1Run['stdClSizes'] = dfClustering1Run['clSizes'].apply(lambda x: np.std(x))

        #dfStdev = dfClustering1Run[['n_clusters', 'stdMedSilScores', 'stdClSizes',
        #                            'silhouetteScore', 'clusterLabels', 'sampleSilhouette']]
        arrPoints = dfClustering1Run[['stdMedSilScores', 'stdClSizes']].to_numpy()
        dfClustering1Run['isPareto'] = is_pareto_efficient(arrPoints)
        dfPareto = dfClustering1Run.loc[dfClustering1Run['isPareto']]
        dfPareto.sort_values(by=['stdMedSilScores'], inplace=True)
        # print('\nPareto points')
        # print(dfPareto[['n_clusters', 'stdMedSilScores', 'stdClSizes', 'isPareto']])
        pList.append(plotClSizesVsMedSilScores())
        plotSilhouetteScoreSamples(dfPareto, runDesc)

    grid = gridplot([[pList[0], pList[1], pList[2]],  # run 1, run 2, run 3
                     [pList[3], pList[4], pList[5]],  # run 4, run 5, run 6
                     [pList[6], pList[7], pList[8]]  # run 7, run 8, run 9
                     ])
    output_file(outFilename, title='Cluster sizes multi runs')
    show(grid)

# ======================================================================================================================
def plotLoglossVersusClSizes(dfLoglossClSizes, clusteringResultsPklFiles, outFilename):
    plotWidth = 250
    plotHeight = 250
    lineWidth = 1.5
    cSize = 7
    alpha = .75

    def plotLogloss():
        p = figure(plot_width=plotWidth, plot_height=plotHeight, x_axis_type="log")
        p.circle(dfLoglossSub['nClusters'], dfLoglossSub['avgLoglossCV'], size=cSize, color='black', alpha=alpha)
        p.line(dfLoglossSub['nClusters'], dfLoglossSub['avgLoglossCV'],
               line_dash='dotted', line_width=lineWidth, color='black', alpha=alpha)
        dfSmallClusters = dfLoglossSub.loc[dfLoglossSub['nClusters'] <= 10]
        p.circle(dfSmallClusters['nClusters'], dfSmallClusters['avgLoglossCV'], size=cSize, color='red', alpha=alpha)
        # adds Label, more info at https://docs.bokeh.org/en/test/docs/user_guide/annotations.html
        runDescLabel = Label(x=5, y=200, x_units='screen', y_units='screen',
                             text=runDesc, render_mode='css',
                             border_line_color='black', border_line_alpha=1.0)
        p.add_layout(runDescLabel)
        return p

    pList = []
    for file, runDesc in clusteringResultsPklFiles.items():
        runId = file.split('_')[1]
        dfLoglossSub = dfLoglossClSizes.loc[dfLoglossClSizes['runId'] == runId]
        dfLoglossSub.sort_values(by=['nClusters'], inplace=True)
        pList.append(plotLogloss())

    grid = gridplot([[pList[0], pList[1], pList[2]],  # run 1, run 2, run 3
                     [pList[3], pList[4], pList[5]],  # run 4, run 5, run 6
                     [pList[6], pList[7], pList[8]]  # run 7, run 8, run 9
                     ])

    output_file(outFilename, title='logloss CV multi runs')
    show(grid)

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
        fig = plt.figure(figsize=(5, 5))
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

        ax.set_title("Sample Silhouette scores of %d clusters (%s)" % (nClusters, runId))
        ax.set_xlabel("Sample Silhouette score")
        ax.set_ylabel("Clusters")
        # The vertical line for average silhouette score of all the values
        ax.axvline(x=avgSilScore, color="red", linestyle="--")
        ax.set_yticks([])  # Clear the yaxis labels / ticks
        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.tight_layout()
        plt.savefig('../myPy/plots/silScoreSamples_%s/%d_Clusters.png' % (runId, nClusters))

# ======================================================================================================================
def plotVarImportance(dfImportance, runId, nClusters, col2Plot, outFilename):
    dfImportance.sort_values(by=[col2Plot], inplace=True, ascending=True)

    plotWidth = 300
    plotHeight = 500
    alpha = .75

    p = figure(plot_width=plotWidth, plot_height=plotHeight, y_range=FactorRange(factors=dfImportance['variable']),
               title='%d clusters (%s)' % (nClusters, runId))
    p.hbar(y=dfImportance['variable'], right=dfImportance[col2Plot], fill_color='black', alpha=alpha)
    output_file(outFilename, title='varSignificant, %d clusters (%s)' % (nClusters, runId))
    show(p)
