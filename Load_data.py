import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

from itertools import chain
# from ravenpackapi import RPApi
# from ravenpackapi import Dataset
import sklearn
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
import glob
import os
import pytz
import datetime
# import pyodbc

# # Packages used for RavenPack
# from ravenpackapi import RPApi
# from ravenpackapi import Dataset
# from ravenpackapi.util import time_intervals, SPLIT_WEEKLY, SPLIT_YEARLY

# # Packages used for connection SQL
# import sqlalchemy
# from sqlalchemy import engine
# from sqlalchemy.engine import URL

from joblib import load


# This function connects and loads historical stock price data from EOD historical
def load_EOD(ar_index):
    print("--- LOADING FROM EOD --- ") 
    
    # File type
    str_format = "json"

    # Initialize df
    df = None

    counter = 0
    for stock in tqdm(ar_index):

        # Creates the URL/API
        str_url_API = "https://eodhistoricaldata.com/api/eod/" + stock + "?api_token=5ffd9842655256.72432671&fmt=" + str_format

        # This try except is because when loading the SP500 index, some stocks are not found (quite a lot actually, +-102)
        try:
            # Makes a dataframe from the json file
            df_stock = pd.read_json(str_url_API)
        except:
            counter += 1
            continue

        # Adds SymbolExchangeCode to the dataframe
        if(df_stock.size > 0):
            df_stock.insert(1,"SymbolExchangeCode", stock)

        # Adjusts column names
        df_stock = df_stock.rename(columns={"adjusted_close":"AdjustedClose"}) 
        df_stock = df_stock.rename(columns={"date":"BarDate"}) 

        # Adds the df_stock created for a single security to the entire df
        df = pd.concat([df, df_stock], ignore_index=True)
    print(f"{counter} stocks were not found in EOD (But were on the list with historical constituents of the S&P500)")
    print("--- END LOADING : EOD FINISHED LOADING --- ")

    return df

# This function creates the correct historical SP500 index
def create_historical_SP_Index_deliver(df):
    # Needed to not change original
    print("len(df) = ", len(df))
    df = df.copy()
    print("len(df) = ", len(df))
    df.drop(columns=['open', 'high', 'low', 'close'], inplace=True)          #, 'volume' is kept
    print("len(df) = ", len(df))

    # Creates string with the URL of the API
    str_url = "https://eodhistoricaldata.com/api/fundamentals/GSPC.INDX?api_token=5ffd9842655256.72432671&historical=1"
    # Creates DF from JSON file
    json_SP = pd.read_json(str_url)
    # This is a dataframe with at each row, a date containing +- 500 dictionaries of the S&P500 constituents
    df2 = json_SP.iloc[870:, 3:]
    data = []
    for date, components in tqdm(df2['HistoricalComponents'].items()):
        # print("date: ", date)
        # print("components: ", components)
        for component in components.values():
            # print("component: ", component)
            data.append({
                'BarDate': pd.to_datetime(component['Date']), 
                'SymbolExchangeCode': component['Code']
            })
    df3 = pd.DataFrame(data)
    print("len(df3) = ", len(df3))

    # Ensure that BarDate columns in both dataframes are of datetime type
    df['BarDate'] = pd.to_datetime(df['BarDate'])
    df3['BarDate'] = pd.to_datetime(df3['BarDate'])

    # delete all rows of df3 and df with BarDate before 2000-01-01 and after 2023-04-21
    df3 = df3[(df3["BarDate"] >= "2000-01-01") & (df3["BarDate"] <= "2023-04-21")]
    df3 = df3.reset_index(drop=True)
    df = df[(df["BarDate"] >= "2000-01-01") & (df["BarDate"] <= "2023-04-21")]
    df = df.reset_index(drop=True)

    print("len(df3) = ", len(df3))  
    print("len(df) = ", len(df))    

    # Plot the number of unique elements of SymbolExchangeCode for every BarDate
    # df3.groupby("BarDate")["SymbolExchangeCode"].nunique().plot()
    # plt.title("Number of unique elements of SymbolExchangeCode for every BarDate")

    # Resample and forward fill within each group of "SymbolExchangeCode"
    df3_daily_list = []
    for symbol in df3['SymbolExchangeCode'].unique():
        df3_symbol = df3[df3['SymbolExchangeCode'] == symbol]
        df3_symbol.set_index('BarDate', inplace=True)  # move set_index operation inside the loop
        df3_daily_symbol = df3_symbol.resample('D').ffill()
        df3_daily_list.append(df3_daily_symbol.reset_index())  # reset index before appending

    # Concatenate all the resampled dataframes back together
    df3_daily = pd.concat(df3_daily_list)
    print("len(df3_daily) = ", len(df3_daily))

    # Rename column to Ticker so they match
    df3_daily = df3_daily.rename(columns={"SymbolExchangeCode":"Ticker"}) 
    print("len(df3_daily) = ", len(df3_daily))
    # Merge df and df3_daily, keeping only rows where 'Ticker' and 'BarDate' match
    df_final = pd.merge(df, df3_daily, how='inner', on=['BarDate', 'Ticker'])
    print("len(df_final) = ", len(df_final))    

    # Plot the number of unique elements of Ticker for every BarDate
    df_final.groupby("BarDate")["Ticker"].nunique().plot()

    # Add title to the plot of the number of unique elements of Ticker for every BarDate
    plt.title("Number of unique elements of Ticker for every BarDate")

    df_final = df_final.sort_values(by=["BarDate", "Ticker"])
    df_final = df_final.reset_index(drop=True)

    return df_final

# This function adds returns to the df
def add_variables(df):
    df = df.copy()
    # Count NaN values per column of df
    print("Number of NaN values per column = \n", df.isna().sum())
    
    # Drop rows with NaN values 
    df = df.dropna().reset_index(drop=True)
    print("NaN value dropped")

    # Create ... lags of "AdjustedClose" for each stock
    lags = 5
    for i in range(1, lags+1):
        df[f'AdjustedClose_Lag_{i}'] = df.groupby('Ticker')['AdjustedClose'].shift(i)
    
    # Create ... leads of "AdjustedClose" for each stock
    leads = 5
    for i in range(1, leads+1):
        df[f'AdjustedClose_Lead_{i}'] = df.groupby('Ticker')['AdjustedClose'].shift(-i)

    # Calculate the 5-day return (percentage) for each stock
    df['FiveDayReturn'] = (df['AdjustedClose'] - df['AdjustedClose_Lag_5']) / df['AdjustedClose_Lag_5'] 

    # Previous day direction is added
    df["PreviousDayDirection"] = (df["AdjustedClose"] > df["AdjustedClose_Lag_1"]).astype(int)

    # Next day direction is added 
    df["NextDayDirection"] = (df["AdjustedClose_Lead_1"] > df["AdjustedClose"]).astype(int)

    # Drop rows with NaN values 
    df = df.dropna().reset_index(drop=True)
    print("NaN value dropped")

    print("df_prices = \n", df.head().to_string())

    # Count NaN values per column of df
    print("Number of NaN values per column = \n", df.isna().sum())
    # Drop rows with NaN values 
    df = df.dropna().reset_index(drop=True)
    print("NaN value dropped")

    # Previous day return is added
    df["PreviousdayReturn"] = (df["AdjustedClose"] - df["AdjustedClose_Lag_1"])/df["AdjustedClose_Lag_1"]

    # Cap outliers
    df = cap_outliers(df,  str_column_name = "PreviousdayReturn")

    # Add previous day returns for 2 and 3 days ago
    df["PreviousdayReturn_2"] = df.groupby('Ticker')['PreviousdayReturn'].shift(1)
    df["PreviousdayReturn_3"] = df.groupby('Ticker')['PreviousdayReturn'].shift(2)

    # Next day return is added
    df["NextdayReturn"] = df.groupby('Ticker')['PreviousdayReturn'].shift(-1).astype(float)
    df["NextdayReturn_2"] = df.groupby('Ticker')['PreviousdayReturn'].shift(-2).astype(float)
    df["NextdayReturn_3"] = df.groupby('Ticker')['PreviousdayReturn'].shift(-3).astype(float)

    # Count NaN values per column of df
    print("Number of NaN values per column = \n", df.isna().sum())
    # Drop rows with NaN values 
    df = df.dropna().reset_index(drop=True)
    print("NaN value dropped")

    # Cross sectional median of next day return
    df["CS_MedianNextdayReturn"] = df.groupby("BarDate")["NextdayReturn"].transform("median")

    # create a new column named Target for df_prices_dayly which is 1 if CS_MedianNextdayReturn < NextdayReturn and 0 otherwise
    df["Target"] = np.where(df["CS_MedianNextdayReturn"] > df["NextdayReturn"], 0, 1)

    # Count the total number of 1's and 0's in the Target column
    print("The total number of 0's and 1's in the target column is: \n", df["Target"].value_counts())

    return df

# Function checks the distribution of returns and caps outliers
def cap_outliers(df, str_column_name):

    # Remove returns larger than 80% and smaller than -80%
    if str_column_name == 'PreviousdayReturn':
        df = df[(df[str_column_name] < 0.75) & (df[str_column_name] > -0.75)]

        print("Unrealistic returns have been removed. \n")

    if str_column_name == 'SESI':
        # Calculate the upper and lower bounds for outliers
        lower_bound = -10
        upper_bound = 10
        print(f"\n Lower bound: {lower_bound} \n Upper bound: {upper_bound} \n")
        df[str_column_name] = df[str_column_name].clip(lower=lower_bound, upper=upper_bound)
        print("\n the large outliers are clipped to -10 and 10 \n")

    # Check the distribution of the following columns of features
    print("The distribution of feature before removing outliers is: \n")
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df[[str_column_name]], bins=200, edgecolor='k', alpha=0.7, color='steelblue')
    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Add title and labels
    plt.title("Distribution of feature")
    plt.xlabel("Variable")
    plt.ylabel("Frequency")
    # Add a vertical line to indicate the mean
    mean = df[str_column_name].mean()
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=2)
    plt.text(mean+0.05, plt.ylim()[1]*0.9, f'Mean: {mean:.2f}', color='red')
    # Add a vertical line to indicate the median
    median = df[str_column_name].median()
    plt.axvline(median, color='green', linestyle='dashed', linewidth=2)
    plt.text(median-0.45, plt.ylim()[1]*0.8, f'Median: {median:.2f}', color='green')
    # Show the histogram
    plt.show()

    # Number of standard deviations to use for the Z-score threshold, 3 is a common value
    threshold = 5
    # Calculate the mean and standard deviation
    # Should actually be calculated on the training set
    mean = df[str_column_name].mean()
    std = df[str_column_name].std()
    # Calculate the upper and lower bounds for outliers
    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std
    print(f"\n Mean = {mean} \n std = {std} \n Lower bound: {lower_bound} \n Upper bound: {upper_bound} \n")
    df[str_column_name] = df[str_column_name].clip(lower=lower_bound, upper=upper_bound)
    print("\n the outliers are clipped \n")

    # Check the distribution of the following columns of features
    print("The distribution of feature after removing outliers is: \n")
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df[[str_column_name]], bins=200, edgecolor='k', alpha=0.7, color='steelblue')
    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Add title and labels
    plt.title("Distribution of feature")
    plt.xlabel("Variable")
    plt.ylabel("Frequency")
    # Add a vertical line to indicate the mean
    mean = df[str_column_name].mean()
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=2)
    plt.text(mean+0.05, plt.ylim()[1]*0.9, f'Mean: {mean:.2f}', color='red')
    # Add a vertical line to indicate the median
    median = df[str_column_name].median()
    plt.axvline(median, color='green', linestyle='dashed', linewidth=2)
    plt.text(median-0.45, plt.ylim()[1]*0.8, f'Median: {median:.2f}', color='green')
    # Show the histogram
    plt.show()

    return df

# This function loads a dataframe from a csv file
def load_csv(file_name):

    # directory = r"C:\Users\BasPeeters\OneDrive - FactorOrange.capital\Master Thesis\Dataframes and output" # Windows
    directory = r"/Users/baspeeters/Library/CloudStorage/OneDrive-FactorOrange.capital/Master Thesis/Dataframes and output" # Macbook
    full_path = os.path.join(directory, f"{file_name}.csv")

    # load csv file
    df = pd.read_csv(full_path)

    return df


def add_SESI(df, df_SESI):
    df = df.copy()
    df_SESI = df_SESI.copy()
    # drop columns RP_ENTITY_ID	ENTITY_NAME	EVENT_SENTIMENT	EVENT_RELEVANCE	EVENT_SIMILARITY_DAYS from df_SESI
    df_SESI.drop(columns=['RP_ENTITY_ID', 'ENTITY_NAME', 'EVENT_SENTIMENT', 'EVENT_RELEVANCE', 'EVENT_SIMILARITY_DAYS'], inplace=True)
    # Rename TimeStamp_TZ to BarDate for df_SESI
    df_SESI = df_SESI.rename(columns = {"TIMESTAMP_TZ": "BarDate"})

    # Merge and make sure the columns are both in datetime
    df_SESI["BarDate"] = pd.to_datetime(df_SESI["BarDate"])
    df["BarDate"] = pd.to_datetime(df["BarDate"])
    df = df.merge(df_SESI, how = "left", on = ["BarDate", "Ticker"])

    # Count all NaN value of every column in df_weekly
    print("There are the following number of NaN values in the new dataframe:")
    print(df.isnull().sum())

    # Turn all NaN values into 0
    df["SESI"] = df["SESI"].fillna(0)
    print("All NaN values are turned into 0")

    # Cap outliers
    df = cap_outliers(df, str_column_name = "SESI")       
    print("Outliers for SESI sare capped")

    return df




# These functions can be deleted later


def create_historical_SP_Index():
    
    # Creating the directory name
    directory = r"C:\Users\BasPeeters\OneDrive - FactorOrange.capital\Master Thesis\Dataframes and output"
    ar_index = load(os.path.join(directory, "stock_universe_new.joblib"))

    df = load_EOD(ar_index)
    print("Loading df worked, delete this line if it worked")
    # del df[["open", "high", "low"]]

    # Creates string with the URL of the API
    str_url = "https://eodhistoricaldata.com/api/fundamentals/GSPC.INDX?api_token=5ffd9842655256.72432671&historical=1"
    # Creates DF from JSON file
    json_SP = pd.read_json(str_url)
    # This is a dataframe with at each row, a date containing +- 500 dictionaries of the S&P500 constituents
    df2 = json_SP.iloc[870:, 3:]
    data = []
    for date, components in tqdm(df2['HistoricalComponents'].items()):
        # print("date: ", date)
        # print("components: ", components)
        for component in components.values():
            # print("component: ", component)
            data.append({
                'BarDate': pd.to_datetime(component['Date']), 
                'SymbolExchangeCode': component['Code']
            })
    df3 = pd.DataFrame(data)

    # Ensure that BarDate columns in both dataframes are of datetime type
    df['BarDate'] = pd.to_datetime(df['BarDate'])
    df3['BarDate'] = pd.to_datetime(df3['BarDate'])

    # delete all rows of df3 and df with BarDate before 2000-01-01 and after 2023-04-21
    df3 = df3[(df3["BarDate"] >= "2000-01-01") & (df3["BarDate"] <= "2023-04-21")]
    df3 = df3.reset_index(drop=True)
    df = df[(df["BarDate"] >= "2000-01-01") & (df["BarDate"] <= "2023-04-21")]
    df = df.reset_index(drop=True)

    # Plot the number of unique elements of SymbolExchangeCode for every BarDate
    df3.groupby("BarDate")["SymbolExchangeCode"].nunique().plot()
    plt.title("Number of unique elements of SymbolExchangeCode for every BarDate")

    # Resample and forward fill within each group of "SymbolExchangeCode"
    df3_daily_list = []
    for symbol in df3['SymbolExchangeCode'].unique():
        df3_symbol = df3[df3['SymbolExchangeCode'] == symbol]
        df3_symbol.set_index('BarDate', inplace=True)  # move set_index operation inside the loop
        df3_daily_symbol = df3_symbol.resample('D').ffill()
        df3_daily_list.append(df3_daily_symbol.reset_index())  # reset index before appending

    # Concatenate all the resampled dataframes back together
    df3_daily = pd.concat(df3_daily_list)

    # Merge df and df3_daily, keeping only rows where 'SymbolExchangeCode' and 'BarDate' match
    df_final = pd.merge(df, df3_daily, how='inner', on=['BarDate', 'SymbolExchangeCode'])

    # Plot the number of unique elements of SymbolExchangeCode for every BarDate
    df_final.groupby("BarDate")["SymbolExchangeCode"].nunique().plot()

    # Add title to the plot of the number of unique elements of SymbolExchangeCode for every BarDate
    plt.title("Number of unique elements of SymbolExchangeCode for every BarDate")

    df_final = df_final.sort_values(by=["BarDate", "SymbolExchangeCode"])
    df_final = df_final.reset_index(drop=True)

    return df_final


# # This function creates and returns a list with the names of a stock index 
# def create_stock_universe_old(str_index_name):
        
#     # Creates string with the URL of the API
#     str_url = "https://eodhistoricaldata.com/api/fundamentals/GSPC.INDX?api_token=5ffd9842655256.72432671"

#     # Creates DF from JSON file
#     json_SP = pd.read_json(str_url)

#     # Creates empty DF which will be filled with the codes of the historical S&P500 
#     df_index = pd.DataFrame(columns=['Code', 'Name', 'StartDate', 'EndDate', 'IsActiveNow', 'IsDelisted'])

#     for i in tqdm(range(10, len(json_SP))):

#         # Selects information for stock i
#         JSON_Stock = json_SP.iloc[i,2]

#         # Makes a dataframe with info for all historical constituents of the S&P500
#         df_index.loc[len(df_index.index)] = JSON_Stock

#     # if (str_index_name == "S&P500_historical"):
#     #     # Creates df with hisrotical codes of SP500
#     #     df_index = df_index[df_index.Code != "SHLD"] # Drops SHLD, this code has no historical market data from EOD

#     if (str_index_name == "S&P500"):
#         # Creates df with all active codes of SP500
#         df_index = df_index[df_index.IsActiveNow == 1]

#     # Creates list with all codes of the specified index
#     ar_index = df_index["Code"].tolist()

#     return ar_index, df_index

# # This function loads data to df from SQL, a query is written
# def load_SQL(str_table_name, str_column_subset):
    
#     print("loading database ...")

#     # Reads an SQL table from Bas.Thesis.Local
#     Database = "Bas.Thesis.Local"
#     Server = "DESKTOP-JF8CK5U" # 10.0.1.6\MAMSTUDIO_DEV

#     # Makes a connection to SQL
#     cnxn = pyodbc.connect("DRIVER={ODBC Driver 17 for SQL Server};"

#                     "Server="+Server+";"  

#                     "Database="+Database+";"

#                     "uid=sa;pwd=sa")

#     # Creates a string which is the query to be read
#     if (str_column_subset == None):
#         query = "SELECT * FROM dbo." +  str_table_name
#     else:
#         query = "SELECT " + str_column_subset + " FROM dbo." +  str_table_name

#     # Reads Sql query
#     df = pd.read_sql_query(sql=query, con=cnxn)

#     # Check if BarDate is a column in df
#     if ("BarDate" in df.columns):

#         # Set column BarDate to datetime
#         df["BarDate"] = pd.to_datetime(df["BarDate"])

#         # Select all rows of df with BarDate from 2000-01-01 onwards
#         df = df[df["BarDate"] >= "2000-01-01"]

#     # # Change name column SymbolExchangeCode of df to Ticker
#     # if ("SymbolExchangeCode" in df.columns):
#     #     df = df.rename(columns={"SymbolExchangeCode":"Ticker"})

#     print("df_prices = \n", df.head().to_string())
#     return df

# # Try this later, should create the actual historical SP500 index
# def create_historical_SP_Index_old_method(df_index, df):

#     # Initialize an empty dataframe with the same columns as df
#     df_filtered = pd.DataFrame(columns=df.columns)

#     # Iterate through df_index
#     for index, row in tqdm(df_index.iterrows()):
#         stock = row["Code"]
#         start_date = row["StartDate"]
#         end_date = row["EndDate"]

#         # Filter df based on stock and the date range
#         filtered_rows = df[(df["SymbolExchangeCode"] == stock) &
#                                     (df["BarDate"] >= start_date) &
#                                     (df["BarDate"] <= end_date)]
#         # Append the filtered rows to the filtered_data dataframe
#         df_filtered = pd.concat([df_filtered, filtered_rows], ignore_index=True)
        
#     return df_filtered









