from bokeh.plotting import figure, output_file, show, save
from bokeh.models import Legend, ColumnDataSource, CustomJS, Range1d, Select
from bokeh.layouts import column, gridplot
from bokeh.transform import dodge
import bokeh.palettes

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from math import pi

# ======================================================================================================================
def plotDemographicCols(dfHts, demoCols, outputFile):
    gridPlots = []

    for col in demoCols:
        if col not in dfHts.columns:
            continue
        dfCounts = dfHts.groupby(col)['indivID'].count().to_frame().rename(columns={'indivID': 'frequency'})
        dfCounts.reset_index(inplace=True)
        countsDict = dfCounts.to_dict(orient='list')
        source = ColumnDataSource(countsDict)
        p = figure(x_range=dfCounts[col].tolist(), plot_width=800, plot_height=500,
                   title='%s counts in 2014 HCMC HTS data' % col, toolbar_location='below')
        p.vbar(x=col, top='frequency', source=source, width=0.9, line_color='white')
        p.x_range.range_padding = 0.1
        p.xaxis.major_label_orientation = pi / 3
        p.xgrid.grid_line_color = None
        gridPlots.append([p])

    # grid = gridplot([[pGender], [pEduLvl]])
    grid = gridplot(gridPlots)
    output_file(outputFile)
    save(grid)


# ======================================================================================================================
'''
def plotDemographicCols_v1(dfHts):
    demoCols = ['gender', 'eduLevel',
                'licenceType', 'ownCar', 'ownMotor', 'ownBike', 'ownEbike', 'travelShareType',
                'isImpaired', 'isHholdHead', 'residencyStatus', 'isHousemaid',
                'jobTitle', 'industry', 'businessType', 'employStatus', 'monthlyIncome']

    p = figure(plot_width=600, plot_height=400,
               title='Demographic attributes in 2014 HCMC HTS data', toolbar_location='below')

    dfGenderCounts = dfHts.groupby('gender')['indivID'].count().to_frame().rename(columns={'indivID': 'frequency'})
    dfGenderCounts.reset_index(inplace=True)
    countsDict = dfGenderCounts.to_dict(orient='list')
    source = ColumnDataSource(countsDict)
    
# ======================================================================================================================
def mkClusteringEvalPlots(silhouetteScores, chScores, nClusters, filename):
    # filename = './tmpOutputs/%s/clusteringSummary.png' % algoStr
    # plots silhouette coefficients
    fig = plt.figure(figsize=[7.5, 5])
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(nClusters, silhouetteScores)# ======================================================================================================================
def mkClusteringEvalPlots(silhouetteScores, chScores, nClusters, filename):
    # filename = './tmpOutputs/%s/clusteringSummary.png' % algoStr
    # plots silhouette coefficients
    fig = plt.figure(figsize=[7.5, 5])
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(nClusters, silhouetteScores)
    ax1.set_xticks(nClusters)
    ax1.set_xlabel("nClusters")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # forces ytick be integers only
    ax1.set_ylabel("Silhouette coefficient")
    ax1.set_ylim([0, ax1.get_ylim()[1]])
    ax1.grid(b=True, linestyle=':', linewidth=1)

    ax2.plot(nClusters, chScores)
    ax2.set_xticks(nClusters)
    ax2.set_xlabel("nClusters")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))  # forces ytick be integers only
    ax2.set_ylabel("Calinski-Harabasz index")
    ax2.set_ylim([0, ax2.get_ylim()[1]])
    ax2.grid(b=True, linestyle=':', linewidth=1)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    ax1.set_xticks(nClusters)
    ax1.set_xlabel("nClusters")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # forces ytick be integers only
    ax1.set_ylabel("Silhouette coefficient")
    ax1.set_ylim([0, ax1.get_ylim()[1]])
    ax1.grid(b=True, linestyle=':', linewidth=1)

    ax2.plot(nClusters, chScores)
    ax2.set_xticks(nClusters)
    ax2.set_xlabel("nClusters")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))  # forces ytick be integers only
    ax2.set_ylabel("Calinski-Harabasz index")
    ax2.set_ylim([0, ax2.get_ylim()[1]])
    ax2.grid(b=True, linestyle=':', linewidth=1)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    #pGender = figure(x_range=dfGenderCounts['gender'].tolist(), plot_width=800, plot_height=400,
    #                 title='Gender counts in 2014 HCMC HTS data', toolbar_location='below')
    pGender = p.vbar(x='gender', top='frequency', source=source, width=0.9, line_color='white')
    print(type(pGender))

    dfEduLvlCounts = dfHts.groupby('eduLevel')['indivID'].count().to_frame().rename(columns={'indivID': 'frequency'})
    dfEduLvlCounts.reset_index(inplace=True)
    countsDict = dfEduLvlCounts.to_dict(orient='list')
    source = ColumnDataSource(countsDict)
    #pEduLvl = figure(x_range=dfEduLvlCounts['eduLevel'].tolist(), plot_width=800, plot_height=400,
    #                 title='Education level counts in 2014 HCMC HTS data', toolbar_location='below')
    pEduLvl = p.vbar(x='eduLevel', top='frequency', source=source, width=0.9, line_color='white')
    print(type(pEduLvl))

    # initialise the plot with only plot1 visible - to match the dropdown default
    pEduLvl.visible = True

    # dropdown widget + Javascript code for interactivity
    select = Select(title="Plot to show:", value="gender", options=["gender", "eduLevel"])
    select.js_on_change("value", CustomJS(args=dict(line_1=pGender, line_2=pEduLvl), code="""
    
    line_1.visible = true
    line_2.visible = true

    if (this.value === "gender") {
        line_2.visible = false 
    } else {
        line_1.visible = false
    }
    
    """))

    layout = column(select, p)
    output_file('demographicColumns.html')
    save(layout)
'''

'''
def plotDemographicCols_v2(dfHts):
    fig = go.Figure()
    demoCols = ['gender', 'eduLevel',
                #'licenceType', 'ownCar', 'ownMotor', 'ownBike', 'ownEbike', 'travelShareType',
                #'isImpaired', 'isHholdHead', 'residencyStatus', 'isHousemaid',
                #'jobTitle', 'industry', 'businessType', 'employStatus', 'monthlyIncome'
                ]
    # makes dataframe of gender s
    dfGenderCounts = dfHts.groupby('gender')['indivID'].count().to_frame().rename(columns={'indivID': 'frequency'})
    dfGenderCounts.reset_index(inplace=True)
    print(dfGenderCounts)
    print(type(dfGenderCounts))
    # makes dataframe of education level counts
    dfEduLvlCounts = dfHts.groupby('eduLevel')['indivID'].count().to_frame().rename(columns={'indivID': 'frequency'})
    dfEduLvlCounts.reset_index(inplace=True)

    fig.add_trace(px.bar(data_frame=dfGenderCounts, x='gender', y='frequency'))
    fig.add_trace(px.bar(data_frame=dfEduLvlCounts, x='eduLevel', y='frequency'))

    buttons = []

    for i,demoCol in enumerate(demoCols):
        args = [False] * len(demoCols)
        args[i] = True
        button = dict(label=demoCol,
                      method="update",
                      args=[{"visible": args}])
        buttons.append(button)

    fig.update_layout(
        updatemenus=[dict(
            active=0,
            type="dropdown",
            buttons=buttons,
            x=0,
            y=1.1,
            xanchor='left',
            yanchor='bottom'
        )],
        autosize=False,
        width=1000,
        height=800
    )

    fig.show()
'''
