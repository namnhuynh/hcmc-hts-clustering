import pandas as pd
import prince
from sklearn.decomposition import PCA
import numpy as np

from bokeh.plotting import figure, output_file, show, save
from bokeh.models import Legend, ColumnDataSource, HoverTool, ColorBar
from bokeh.io import push_notebook, show, output_notebook
from bokeh.layouts import gridplot
from bokeh.transform import linear_cmap
from bokeh.palettes import d3  # Turbo256 #Spectral6

# ======================================================================================================================
def doPCA(dfHTS):
    tripAttribTypes = ['destTrip', #'typeDestTrip',
                       'purposeTrip', 'modeTrip', 'nMinsTrip', 'nMinsDepTimeTrip']
    tripAttribCols = []
    for iTrip in range(1, 7 + 1):  # for each trip from 1 to 7
        for attrib in tripAttribTypes:
            tripAttribCols.append('%s%d' % (attrib, iTrip))
    dfHTSTrips = dfHTS[tripAttribCols]

    # one hot encodes categorical attributes
    tripAttribs = ['origTrip', 'destTrip', 'typeOrigTrip', 'typeDestTrip', 'purposeTrip', 'modeTrip']
    for tripAttrib in tripAttribs:
        for iTrip in range(1, 7 + 1):  # for each trip from 1 to 7
            dummies = pd.get_dummies(dfHTSTrips['%s%d' % (tripAttrib, iTrip)], prefix='%s%d' % (tripAttrib, iTrip))
            dfHTSTrips = dfHTSTrips.join(dummies)
            dfHTSTrips.drop(columns=['%s%d' % (tripAttrib, iTrip)], inplace=True)
    dfHTSTrips.to_csv('./tmpOutputs/dfHTSTrips_ohe0.csv', index=False)

    # standardises variables
    def standardise(df):
        dfStd = pd.DataFrame()
        for col in df.columns:
            dfStd[col] = (df[col] - df[col].mean()) / df[col].std()
        return dfStd
    dfHTSTripsStd = standardise(dfHTSTrips)

    # calculates the principal components
    nPCs = len(dfHTSTripsStd.columns.tolist())
    pcColumns = ['PC%d' % i for i in range(1, nPCs + 1)]
    pca = PCA(n_components=len(pcColumns))
    pca.fit(dfHTSTripsStd)

    # produces the scree plot
    print('ratio of variance explained')
    print(pca.explained_variance_ratio_)
    print('total variance explained')
    print(np.cumsum(pca.explained_variance_ratio_))

    def mkScreePlots(pcaResults):
        dfExplainedVar = pd.DataFrame({'ratioVarExp': pcaResults.explained_variance_ratio_,
                                       'totalVarExp': np.cumsum(pca.explained_variance_ratio_),
                                       'principalComp':
                                           [i + 1 for i in range(len(pcaResults.explained_variance_ratio_))]})
        legendList = []
        p = figure(plot_width=800, plot_height=400,
                   title='variance explained of principal components')
        line = p.line(x='principalComp', y='ratioVarExp', source=ColumnDataSource(dfExplainedVar))
        circles = p.circle(x='principalComp', y='ratioVarExp', source=ColumnDataSource(dfExplainedVar), size=4)
        bars = p.vbar(x='principalComp', top='totalVarExp', width=.9, source=ColumnDataSource(dfExplainedVar),
                      alpha=.4)
        legendList.append(('variance explained ratio', [line, circles]))
        legendList.append(('total variance explained', [bars]))
        # formats RSS plot
        p.xaxis.axis_label = 'principal component'
        p.yaxis.axis_label = 'variation explained'
        legend = Legend(items=legendList)
        legend.click_policy = 'hide'
        p.add_layout(legend, 'right')
        output_file('./tmpOutputs/pcaScreePlot.html', title='mobility and f0')
        show(p)

    mkScreePlots(pca)

# ======================================================================================================================
def doFAMD(dfHTS):
    tripAttribTypes = ['destTrip', #'typeDestTrip',
                       'purposeTrip', 'modeTrip', 'nMinsTrip', 'nMinsDepTimeTrip']
    tripAttribCols = []
    for iTrip in range(1, 7 + 1):  # for each trip from 1 to 7
        for attrib in tripAttribTypes:
            tripAttribCols.append('%s%d' % (attrib, iTrip))

    famd = prince.FAMD(n_components=len(tripAttribCols),  # n_components=4
                       copy=True, engine='auto', random_state=0)
    famd = famd.fit(dfHTS[tripAttribCols])

    print('famd eigenvalues')
    print(famd.eigenvalues_)
    print('famd explained_inertia_')
    print(famd.explained_inertia_)

    print('\nfamd column_correlations')
    colCorrelations = famd.column_correlations(dfHTS[tripAttribCols])
    print(type(colCorrelations))
    print(colCorrelations.shape)

    print('\nrow_coordinates')
    dfTripsFAMD = famd.row_coordinates(dfHTS[tripAttribCols])
    print(type(dfTripsFAMD))
    print(dfTripsFAMD.shape)

    # adds indivID to dfTripsFAMD
    dfTripsFAMD['indivID'] = dfHTS['hhID'].astype(str) + '_' + dfHTS['indivID'].astype(str)

    return colCorrelations, dfTripsFAMD
