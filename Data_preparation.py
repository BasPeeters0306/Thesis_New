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

    if "SESI_lagged" in df.columns:
        # Check the distribution of the following columns of features
        print("The distribution of SESI_lagged before scaling is is: \n")
        # Create histogram
        plt.figure(figsize=(10, 6))
        plt.hist(df[["SESI_lagged"]], bins=200, edgecolor='k', alpha=0.7, color='steelblue')
        # Add grid lines
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        # Add title and labels
        plt.title("Distribution of SESI_lagged")
        plt.xlabel("Variable")
        plt.ylabel("Frequency")
        # Add a vertical line to indicate the mean
        mean = df["SESI_lagged"].mean()
        plt.axvline(mean, color='red', linestyle='dashed', linewidth=2)
        plt.text(mean+0.05, plt.ylim()[1]*0.9, f'Mean: {mean:.2f}', color='red')
        # Add a vertical line to indicate the median
        median = df["SESI_lagged"].median()
        plt.axvline(median, color='green', linestyle='dashed', linewidth=2)
        plt.text(median-0.45, plt.ylim()[1]*0.8, f'Median: {median:.2f}', color='green')
        # Show the histogram
        plt.show()

    if "SESI_lagged" not in df.columns:
        print("SESI_lagged is not in df_dayly")
        # Scales features
        if b_MinMaxScaler == True:
            scaler = MinMaxScaler((-1,1)) #MinMaxScaler((-1,1))
            scaler.fit(df[["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3"]])
            df[["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3"]] = scaler.transform(df[["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3"]])
        elif b_standardizer == True:
            scaler_std = StandardScaler()
            scaler_std.fit(df[["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3"]])
            df[["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3"]] = scaler_std.transform(df[["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3"]])
    
    if "SESI_lagged" in df.columns:
        print("SESI_lagged is in df_dayly and will also be scaled")
        # Scales features
        if b_MinMaxScaler == True:
            scaler = MinMaxScaler((-1,1))   #MinMaxScaler((-1,1))
            # We scale based on values in the training set
            scaler.fit(df.iloc[:split_number][["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3", "SESI_lagged"]])
            df[["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3", "SESI_lagged"]] = scaler.transform(df[["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3", "SESI_lagged"]])
        elif b_standardizer == True:
            scaler_std = StandardScaler()
            # We scale based on values in the training set
            scaler_std.fit(df.iloc[:split_number][["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3", "SESI_lagged"]])
            df[["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3", "SESI_lagged"]] = scaler_std.transform(df[["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3", "SESI_lagged"]])

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

    if "SESI_lagged" in df.columns:
        # Check the distribution of the following columns of features
        print("The distribution of SESI_lagged is: \n")
        # Create histogram
        plt.figure(figsize=(10, 6))
        plt.hist(df[["SESI_lagged"]], bins=200, edgecolor='k', alpha=0.7, color='steelblue')
        # Add grid lines
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        # Add title and labels
        plt.title("Distribution of SESI_lagged")
        plt.xlabel("Variable")
        plt.ylabel("Frequency")
        # Add a vertical line to indicate the mean
        mean = df["SESI_lagged"].mean()
        plt.axvline(mean, color='red', linestyle='dashed', linewidth=2)
        plt.text(mean+0.05, plt.ylim()[1]*0.9, f'Mean: {mean:.2f}', color='red')
        # Add a vertical line to indicate the median
        median = df["SESI_lagged"].median()
        plt.axvline(median, color='green', linestyle='dashed', linewidth=2)
        plt.text(median-0.45, plt.ylim()[1]*0.8, f'Median: {median:.2f}', color='green')
        # Show the histogram
        plt.show()

    return df

# Function creates splits and features and target dataframes
def create_splits(df, percentage_test_set, model, b_sentiment_score):   

    if model == "lstm":
        if b_sentiment_score == True:
            if "SESI_lagged" in df.columns:
                # Creates feature dataframe
                X = df.loc[:, df.columns.isin(["BarDate", "Ticker", "PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3", "SESI_lagged"])] #"BarDate", "Ticker", 
        else:
            # Creates feature dataframe
            X = df.loc[:, df.columns.isin(["BarDate", "Ticker", "PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3"])] #"BarDate", "Ticker", 

        # Creates target dataframe
        y = df.loc[:, df.columns.isin(["BarDate", "Ticker", "Target"])] #"BarDate", "Ticker", 

    if model == "not_lstm":
        if b_sentiment_score == True:
            if "SESI_lagged" in df.columns:
                # Creates feature dataframe
                X = df.loc[:, df.columns.isin(["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3", "SESI_lagged"])] #"BarDate", "Ticker", 
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


