import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm._tqdm_notebook import tqdm_notebook
tqdm_notebook.pandas()


# Creates a daily SESI Score from raw RavenPack data. 
def SESI(df):
    df = df.copy()	

    # Turn time to a daily frequency
    df['TIMESTAMP_TZ'] = df['TIMESTAMP_TZ'].astype(str).str[:10]

    # Relevant filters for SESI
    df = df[(df['EVENT_SENTIMENT'] != 0)]
    df = df[(df['EVENT_RELEVANCE'] >= 90) & (df['EVENT_SIMILARITY_DAYS'] >= 90)]

    # Calculates the daily bias
    df['F1'] = df['EVENT_SENTIMENT'] * ((df['EVENT_RELEVANCE'] / 100)**2) * ((df['EVENT_SIMILARITY_DAYS'] / 365)**2)
    df['F2'] = ((df['EVENT_RELEVANCE'] / 100)**2) * ((df['EVENT_SIMILARITY_DAYS'] / 365)**2)
    df['SumF1'] = df.groupby(df['TIMESTAMP_TZ'])['F1'].transform('sum')
    df['SumF2'] = df.groupby(df['TIMESTAMP_TZ'])['F2'].transform('sum')
    df['DAILY_BIAS'] = df['SumF1'] / df['SumF2']

    # Subtracts the daily bias and sums over all events on the date for the entity to get SESI
    df['ESS - DAILY_BIAS'] = df['EVENT_SENTIMENT'] - df['DAILY_BIAS']
    df['SESI'] = df.groupby(['TIMESTAMP_TZ', 'RP_ENTITY_ID'])['ESS - DAILY_BIAS'].transform('sum')
    
    # Drop duplicates, only one SESI er day per entity needed
    df['unique'] = df['TIMESTAMP_TZ'] + df['RP_ENTITY_ID']
    df = df.drop_duplicates(subset = ['unique'])

    # Drop irrelevant columns
    df.drop(columns=['F1', 'F2', 'SumF1', 'SumF2', 'DAILY_BIAS', "ESS - DAILY_BIAS", "unique"], inplace=True)

    return df

# Adds Ticker to df_SESI
def add_ticker(df_SESI):
    
    # del df_SESI['Unnamed: 0']
    df_SESI_ticker = df_SESI.reset_index()
    df_SESI_ticker['Ticker'] = ""

    # Reads Constituents_SP500 csv file
    df_constituents =  pd.read_csv('Constituents_SPH')

    # Adds ticker to df_SESI
    for i in tqdm(range(0,len(df_SESI_ticker))):
        try:
            row = df_constituents.index[df_SESI_ticker['RP_ENTITY_ID'].loc[i] == df_constituents['RP_ENTITY_ID']].tolist()
            df_SESI_ticker.loc[i, 'Ticker'] = df_constituents['Code'].loc[row].tolist()[-1]   
        except:
            continue

    return df_SESI_ticker


    
    





