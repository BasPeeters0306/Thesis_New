import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt



def feature_scaling(df, b_MinMaxScaler, b_standardizer, percentage_test_set):

    split_number = int(percentage_test_set * len(df))

    # Check the distribution of the following columns of features
    print("The distribution of PreviousdayReturn before scaling is: \n")
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df[["PreviousdayReturn"]], bins=200, edgecolor='k', alpha=0.7, color='steelblue')
    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Add title and labels
    plt.title("Distribution of PreviousdayReturn")
    plt.xlabel("Variable")
    plt.ylabel("Frequency")
    # Add a vertical line to indicate the mean
    mean = df["PreviousdayReturn"].mean()
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=2)
    plt.text(mean+0.05, plt.ylim()[1]*0.9, f'Mean: {mean:.2f}', color='red')
    # Add a vertical line to indicate the median
    median = df["PreviousdayReturn"].median()
    plt.axvline(median, color='green', linestyle='dashed', linewidth=2)
    plt.text(median-0.45, plt.ylim()[1]*0.8, f'Median: {median:.2f}', color='green')
    # Show the histogram
    plt.show()

    if "SESI" in df.columns:
        # Check the distribution of the following columns of features
        print("The distribution of SESI before scaling is is: \n")
        # Create histogram
        plt.figure(figsize=(10, 6))
        plt.hist(df[["SESI"]], bins=200, edgecolor='k', alpha=0.7, color='steelblue')
        # Add grid lines
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        # Add title and labels
        plt.title("Distribution of SESI")
        plt.xlabel("Variable")
        plt.ylabel("Frequency")
        # Add a vertical line to indicate the mean
        mean = df["SESI"].mean()
        plt.axvline(mean, color='red', linestyle='dashed', linewidth=2)
        plt.text(mean+0.05, plt.ylim()[1]*0.9, f'Mean: {mean:.2f}', color='red')
        # Add a vertical line to indicate the median
        median = df["SESI"].median()
        plt.axvline(median, color='green', linestyle='dashed', linewidth=2)
        plt.text(median-0.45, plt.ylim()[1]*0.8, f'Median: {median:.2f}', color='green')
        # Show the histogram
        plt.show()

    if "SESI" not in df.columns:
        print("SESI is not in df_dayly")
        # Scales features
        if b_MinMaxScaler == True:
            scaler = MinMaxScaler((0,1)) #MinMaxScaler((-1,1))
            scaler.fit(df[["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3"]])
            df[["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3"]] = scaler.transform(df[["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3"]])
        elif b_standardizer == True:
            scaler_std = StandardScaler()
            scaler_std.fit(df[["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3"]])
            df[["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3"]] = scaler_std.transform(df[["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3"]])
    
    if "SESI" in df.columns:
        print("SESI is in df_dayly and will also be scaled")
        # Scales features
        if b_MinMaxScaler == True:
            scaler = MinMaxScaler((0,1))   #MinMaxScaler((-1,1))
            # We scale based on values in the training set
            scaler.fit(df.iloc[:split_number][["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3", "SESI"]])
            df[["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3", "SESI"]] = scaler.transform(df[["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3", "SESI"]])
        elif b_standardizer == True:
            scaler_std = StandardScaler()
            # We scale based on values in the training set
            scaler_std.fit(df.iloc[:split_number][["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3", "SESI"]])
            df[["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3", "SESI"]] = scaler_std.transform(df[["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3", "SESI"]])

    # Check the distribution of the following columns of features
    print("The distribution PreviousdayReturn is: \n")
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df[["PreviousdayReturn"]], bins=200, edgecolor='k', alpha=0.7, color='steelblue')
    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Add title and labels
    plt.title("Distribution of PreviousdayReturn")
    plt.xlabel("Variable")
    plt.ylabel("Frequency")
    # Add a vertical line to indicate the mean
    mean = df["PreviousdayReturn"].mean()
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=2)
    plt.text(mean+0.05, plt.ylim()[1]*0.9, f'Mean: {mean:.2f}', color='red')
    # Add a vertical line to indicate the median
    median = df["PreviousdayReturn"].median()
    plt.axvline(median, color='green', linestyle='dashed', linewidth=2)
    plt.text(median-0.45, plt.ylim()[1]*0.8, f'Median: {median:.2f}', color='green')
    # Show the histogram
    plt.show()

    if "SESI" in df.columns:
        # Check the distribution of the following columns of features
        print("The distribution of SESI is: \n")
        # Create histogram
        plt.figure(figsize=(10, 6))
        plt.hist(df[["SESI"]], bins=200, edgecolor='k', alpha=0.7, color='steelblue')
        # Add grid lines
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        # Add title and labels
        plt.title("Distribution of SESI")
        plt.xlabel("Variable")
        plt.ylabel("Frequency")
        # Add a vertical line to indicate the mean
        mean = df["SESI"].mean()
        plt.axvline(mean, color='red', linestyle='dashed', linewidth=2)
        plt.text(mean+0.05, plt.ylim()[1]*0.9, f'Mean: {mean:.2f}', color='red')
        # Add a vertical line to indicate the median
        median = df["SESI"].median()
        plt.axvline(median, color='green', linestyle='dashed', linewidth=2)
        plt.text(median-0.45, plt.ylim()[1]*0.8, f'Median: {median:.2f}', color='green')
        # Show the histogram
        plt.show()

    return df

# Function creates splits and features and target dataframes
def create_splits(df, percentage_test_set, model, b_sentiment_score):   

    if model == "lstm":
        if b_sentiment_score == True:
            if "SESI" in df.columns:
                # Creates feature dataframe
                X = df.loc[:, df.columns.isin(["BarDate", "Ticker", "PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3", "SESI"])] #"BarDate", "Ticker", 
        else:
            # Creates feature dataframe
            X = df.loc[:, df.columns.isin(["BarDate", "Ticker", "PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3"])] #"BarDate", "Ticker", 

        # Creates target dataframe
        y = df.loc[:, df.columns.isin(["BarDate", "Ticker", "Target"])] #"BarDate", "Ticker", 

    if model == "not_lstm":
        if b_sentiment_score == True:
            if "SESI" in df.columns:
                # Creates feature dataframe
                X = df.loc[:, df.columns.isin(["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3", "SESI"])] #"BarDate", "Ticker", 
        else:
            # Creates feature dataframe
            X = df.loc[:, df.columns.isin(["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3"])] #"BarDate", "Ticker", 

        # Creates target dataframe
        y = df.loc[:, df.columns.isin(["Target"])] #"BarDate", "Ticker", 

    # Creates a in ssample and test set # 60-40 split, this includes fin crisis
    split_number = int(percentage_test_set * len(df))

    X_in_sample = X.iloc[:split_number]
    X_test = X.iloc[split_number:]
    y_in_sample = y.iloc[:split_number]
    y_test = y.iloc[split_number:]

    print("Number of obs in sample set = ", len(X_in_sample))
    print("Number of obs test set = ", len(X_test))

    print("Last 5 columns of X_in_sample: \n", X_in_sample.tail().to_string())
    print("First 5 columns of X_test: \n", X_test.head().to_string())
    
    # Creates a train and val set # 80-20 split
    split_number = int(0.8 * len(X_in_sample))

    X_train = X_in_sample.iloc[:split_number]
    X_val = X_in_sample.iloc[split_number:]
    y_train = y_in_sample.iloc[:split_number]
    y_val = y_in_sample.iloc[split_number:]

    print("Number of obs train set = ", len(X_train))
    print("Number of obs val set = ", len(X_val))

    print("Last 5 columns of X_train: \n", X_train.tail().to_string())
    print("First 5 columns of X_val: \n", X_val.head().to_string())

    return X_in_sample, X_test, y_in_sample, y_test, X_train, X_val, y_train, y_val

# # This function creates new variables based on historical stock price data
# Not used?
# def load_new_variables(df):
    
#     # Count NaN values per column of df
#     print("Number of NaN values = ", df.isna().sum())
#     print("Number of null values = ", df.isnull().sum())

#     # Drop rows with NaN values 
#     df = df.dropna().reset_index(drop=True)

#     # Drops rows with NaN and checks if all rows with NaN are deleted
#     df = df.dropna()
#     df = df.sort_values("BarDate")
   
#     # Past day stock growth is added
#     df["PastdayGrowth"] = (df["AdjustedClose"] - df["PreviousAdjustedClose"]) / df["PreviousAdjustedClose"] 
#     #df["Past2dayGrowth"] = (df["AdjustedClose"] - df["Previous2AdjustedClose"]) / df["Previous2AdjustedClose"] 
#     df["Past3dayGrowth"] = (df["AdjustedClose"] - df["Previous3AdjustedClose"]) / df["Previous3AdjustedClose"]
#     df["Past5dayGrowth"] = (df["AdjustedClose"] - df["Previous5AdjustedClose"]) / df["Previous5AdjustedClose"]
#     # df["Past7dayGrowth"] = (df["AdjustedClose"] - df["Previous7AdjustedClose"]) / df["Previous7AdjustedClose"]
#     # df["Past30dayGrowth"] = (df["AdjustedClose"] - df["Previous30AdjustedClose"]) / df["Previous30AdjustedClose"]
#     # df["Past90dayGrowth"] = (df["AdjustedClose"] - df["Previous90AdjustedClose"]) / df["Previous90AdjustedClose"]
#     df["Past21dayGrowth"] = (df["AdjustedClose"] - df["Previous21AdjustedClose"]) / df["Previous21AdjustedClose"]
    
#     # Next day stock growth is added
#     df["NextdayGrowth"] = (df["NextAdjustedClose"] - df["AdjustedClose"]) / df["AdjustedClose"]
#     #df["Next2dayGrowth"] = (df["Next2AdjustedClose"] - df["AdjustedClose"]) / df["AdjustedClose"]
#     df["Next3dayGrowth"] = (df["Next3AdjustedClose"] - df["AdjustedClose"]) / df["AdjustedClose"]
#     df["Next5dayGrowth"] = (df["Next5AdjustedClose"] - df["AdjustedClose"]) / df["AdjustedClose"]
#     # df["Next7dayGrowth"] = (df["Next7AdjustedClose"] - df["AdjustedClose"]) / df["AdjustedClose"]
#     # df["Next30dayGrowth"] = (df["Next30AdjustedClose"] - df["AdjustedClose"]) / df["AdjustedClose"]
#     # df["Next90dayGrowth"] = (df["Next90AdjustedClose"] - df["AdjustedClose"]) / df["AdjustedClose"]
#     df["Next21dayGrowth"] = (df["Next21AdjustedClose"] - df["AdjustedClose"]) / df["AdjustedClose"]

#     # Previous day direction is added
#     df["PreviousdayDirection"] = (df["AdjustedClose"] > df["PreviousAdjustedClose"]).astype(int)

#     # Next day direction is added 
#     df["NextdayDirection"] = (df["NextAdjustedClose"] > df["AdjustedClose"]).astype(int)
    
#     return df

