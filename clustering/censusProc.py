import pandas as pd
import os.path
import pickle
import pyreadstat

pd.set_option('display.max_columns', None)

censusHholdV_sav = '../../../data/census/Census2019_Ban ghi ho.sav'
censusHholdV_pkl = '../../../data/census/Census2019_Ban ghi ho.pkl'
censusIndV_sav = '../../../data/census/Census2019_Ban ghi nguoi.sav'
censusIndV_pkl = '../../../data/census/Census2019_Ban ghi nguoi.pkl'

def preprocessCensus(dfCensus):
    print(dfCensus.head(10))
    print(list(dfCensus))

def readCensusData(census_sav, census_pkl):
    if os.path.exists(census_pkl):
        return pickle.load(census_pkl)
    else:
        dfCensus = pd.read_spss(census_sav)
        print(dfCensus.head(10))
        #dfCensus.to_pickle(census_pkl)
        return dfCensus

def readIndivCensusData(census_sav):
    dfCensus = pyreadstat.pyreadstat.read_sav(filename_path=census_sav)
    return dfCensus
