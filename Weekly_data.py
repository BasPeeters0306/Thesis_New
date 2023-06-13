import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore


def is_wednesday(df):
    return df[df['WeekDay']==2].shape[0]

def is_tuesday(df):
    return df[df['WeekDay']==1].shape[0]

def is_monday(df):
    return df[df['WeekDay']==0].shape[0]

# def is_thursday(df):
#     return df[df['WeekDay']==3].shape[0]     # This is wrong, should be monday!!!!!!!!!!!

# if rebelancing is middle of the week
# returns dates
def get_middle_of_weekdates(df):

    # group on dates
    df_grouped = df.groupby(by='BarDate').sum().reset_index()

    # add year and week/weeknumber columns
    df_grouped['WeekDay'] = df_grouped['BarDate'].dt.day_of_week
    df_grouped['WeekNumber'] = df_grouped['BarDate'].dt.isocalendar().week
    df_grouped['Year'] = df_grouped['BarDate'].dt.isocalendar().year


    df_grouped = df_grouped[['BarDate', 'WeekDay', 'WeekNumber', 'Year']]
    years = df_grouped['Year'].unique()
    weeks = df_grouped['WeekNumber'].unique()

    dates = []

    # loop through years
    for year in years:
        for weeknumber in weeks:
            days = df_grouped[(df_grouped['Year'] == year) & (df_grouped['WeekNumber'] == weeknumber)]

            # if no days continue
            if days.empty:
                continue

            if is_wednesday(days):
                dates.append(days[days['WeekDay']==2]['BarDate'].values[-1])
                continue

            if is_tuesday(days):
                dates.append(days[days['WeekDay']==1]['BarDate'].values[-1])
                continue
            
            if is_monday(days):
                dates.append(days[days['WeekDay']==0]['BarDate'].values[-1])
                continue
    
    dates_shift = np.roll(dates, -1)

    output = [(dates[i], dates_shift[i]) for i in range(len(dates))]
     
    return output[:len(output)-2]

def create_weekly_df(df, dates):
    # format output
    output_cols = ['OpenDate', 'Ticker', 'OpenPrice', 'ClosePrice', 'Direction', 'CloseDate']
    output_df = pd.DataFrame(data=[], columns=output_cols)

    # loop through all dates
    for first_date, last_date in dates:
        
        # calculate difference in days
        diff = last_date - first_date
        days = diff.astype('timedelta64[D]')
        days = int(days / np.timedelta64(1, 'D'))-2

        if days > 5:
            continue

        # get stocks
        week_stocks = df[df['BarDate'] == first_date]

        # check for which next close
        if days < 2:
            closing_col = f"AdjustedClose_Lead_1"
        else:
            closing_col = f"AdjustedClose_Lead_{days}"

        # select correct columns and add to output
        cols = ['BarDate', 'SymbolExchangeCode', 'AdjustedClose', closing_col, 'NextDayDirection']
        chosen_stocks = week_stocks[cols]
        chosen_stocks.loc[:, 'SellDate'] = last_date
        chosen_stocks.columns = output_cols
        output_df = pd.concat([output_df, chosen_stocks], ignore_index=True)
        #output_df = output_df.append(chosen_stocks, ignore_index=True)

    # convert to correct dtypes
    output_df = output_df.convert_dtypes()

    return output_df

# Function checks the distribution of returns and caps outliers
def cap_outliers(df, str_column_name):

    # Remove weekly returns larger than 100% and smaller than -100%
    if str_column_name == 'PreviousWeekReturn':
        df = df[(df[str_column_name] < 1) & (df[str_column_name] > -1)]

    print("Unrealistic returns have been removed. \n")

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
    threshold = 3
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

# This function adds returns to the df
def add_lags(df):

    # Count NaN values per column of df
    print("Number of NaN values per column = \n", df.isna().sum())
    
    # Drop rows with NaN values 
    df = df.dropna().reset_index(drop=True)
    print("NaN value dropped")
    
    # Drop all rows of df["ClosePriceLag"] == 0
    df = df[df["ClosePrice"] != 0].reset_index(drop=True)

    # Create ... lags of "AdjustedClose" for each stock
    lags = 1
    for i in range(1, lags+1):
        df[f'ClosePriceLag_{i}'] = df.groupby('Ticker')['ClosePrice'].shift(i).astype(float)
    
    # Create ... leads of "AdjustedClose" for each stock
    leads = 1
    for i in range(1, leads+1):
        df[f'ClosePriceLead_{i}'] = df.groupby('Ticker')['ClosePrice'].shift(-i).astype(float)

    # Count NaN values per column of df
    print("Number of NaN values per column = \n", df.isna().sum())
    # Drop rows with NaN values 
    df = df.dropna().reset_index(drop=True)
    print("NaN value dropped")

    # Previous week return is added
    df["PreviousWeekReturn"] = (df["ClosePrice"] - df["ClosePriceLag_1"])/df["ClosePriceLag_1"]
    
    # Cap outliers
    df = cap_outliers(df,  str_column_name = "PreviousWeekReturn")

    # Add previous week returns for 2 and 3 weeks ago
    df["PreviousWeekReturn_2"] = df.groupby('Ticker')['PreviousWeekReturn'].shift(1)
    df["PreviousWeekReturn_3"] = df.groupby('Ticker')['PreviousWeekReturn'].shift(2)

    # Count NaN values per column of df
    print("Number of NaN values per column = \n", df.isna().sum())
    # Drop rows with NaN values 
    df = df.dropna().reset_index(drop=True)
    print("NaN value dropped")

    # Next week return is added
    df["NextWeekReturn"] = df.groupby('Ticker')['PreviousWeekReturn'].shift(-1).astype(float)

    # Count NaN values per column of df
    print("Number of NaN values per column = \n", df.isna().sum())
    # Drop rows with NaN values 
    df = df.dropna().reset_index(drop=True)
    print("NaN value dropped")

    # Previous day direction is added
    df["PreviousWeekDirection"] = (df["ClosePrice"] > df["ClosePriceLag_1"]).astype(int)

    # Next day direction is added 
    df["NextWeekDirection"] = (df["ClosePriceLead_1"] > df["ClosePrice"]).astype(int)

    # Cross sectional median of next day return
    df["CS_MedianNextWeekReturn"] = df.groupby("CloseDate")["NextWeekReturn"].transform("median")

    # create a new column named Target for df_prices_weekly which is 1 if CS_MedianNextWeekReturn < NextWeekReturn and 0 otherwise
    df["Target"] = np.where(df["CS_MedianNextWeekReturn"] > df["NextWeekReturn"], 0, 1)

    # Count the total number of 1's and 0's in the Target column
    print("The total number of 0's and 1's in the target column is: \n", df["Target"].value_counts())

    df = df.sort_values("OpenDate").reset_index(drop=True)

    print("df_prices = \n", df.head().to_string())
    return df