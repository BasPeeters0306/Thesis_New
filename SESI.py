import numpy as np
import pandas as pd
from tqdm.auto import tqdm
# from tqdm import tqdm
# from tqdm._tqdm_notebook import tqdm_notebook
# tqdm_notebook.pandas()




# This function creates a daily SESI Score from the raw RavenPack data. Change this function!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Same as WAZNI now!!!!!!!!! Do this later!!!!!!!!!!!!!!!!!!!
def df_SESI(df):

    # Filter columns
    df = df[(df['EVENT_RELEVANCE'] >= 90) & (df['EVENT_SIMILARITY_DAYS'] >= 90)]

    # Rename column
    df = df.rename(columns = {'timestamp_tz' : 'TIMESTAMP_TZ'})

    # Cut the timestamp such that it is on a daily frequency
    df['TIMESTAMP_TZ'] = df['TIMESTAMP_TZ'].astype(str).str[:10]

    # First do calculations over all the granular events
    df['functionColumn1'] = df['EVENT_SENTIMENT'] * ((df['EVENT_RELEVANCE'] / 100)**2) * ((df['EVENT_SIMILARITY_DAYS'] / 365)**2)
    df['functionColumn2'] = ((df['EVENT_RELEVANCE'] / 100)**2) * ((df['EVENT_SIMILARITY_DAYS'] / 365)**2)

    # Make new columns of the sums of the above columns grouped by each day
    df['SumFunctionColumn1'] = df.groupby(df['TIMESTAMP_TZ'])['functionColumn1'].transform('sum')
    df['SumFunctionColumn2'] = df.groupby(df['TIMESTAMP_TZ'])['functionColumn2'].transform('sum')

    # Take the difference of the above two columns to get the daily bias
    df['DAILY_BIAS'] = df['SumFunctionColumn1'] / df['SumFunctionColumn2']

    # Now take the the difference of the ESS and the daily bias 
    df['ESS - DAILY_BIAS'] = df['EVENT_SENTIMENT'] - df['DAILY_BIAS']

    # Now take the sum based on the date and the entity to obtain the SESI
    df['SESI'] = df.groupby(['TIMESTAMP_TZ', 'RP_ENTITY_ID'])['ESS - DAILY_BIAS'].transform('sum')
    
    # Drop duplicates: we only need one SESI per day per entity. Note: this means that the only remaining valid columns are date, entity, and SESI as we drop a lot of rows 
    df['unique'] = df['TIMESTAMP_TZ'] + df['RP_ENTITY_ID']
    df = df.drop_duplicates(subset = ['unique'])

    # Drop irrelevant columns
    df.drop(columns=['functionColumn1', 'functionColumn2', 'SumFunctionColumn1', 'SumFunctionColumn2', 'DAILY_BIAS', "ESS - DAILY_BIAS", "unique"], inplace=True)

    return df





    





