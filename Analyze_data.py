import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from tqdm import tqdm
import matplotlib.dates as mdates


def Data_analysis(df, bool_dict):
    
    # Count NaN values per column of df
    print("NaN values per column: \n", df.isna().sum())

    # Initialze dictionary to store analysis results
    analyze_dict = {}

    if (bool_dict['descriptive_stats'] == True):
        analyze_dict["descriptive_stats"] = df.describe()
        print('descriptive_stats', analyze_dict["descriptive_stats"])

    if (bool_dict['correlation_matrix'] == True):
        analyze_dict["correlation_matrix"] = df.corr()
        print('correlation_matrix', analyze_dict["correlation_matrix"])

    if (bool_dict['histogram'] == True):
        analyze_dict["histogram"] = df.hist()
        print('histogram', analyze_dict["histogram"])
    
    if (bool_dict['boxplot'] == True):
        analyze_dict["boxplot"] = df.boxplot()
        print('boxplot', analyze_dict["boxplot"])

    if (bool_dict['scatterplot'] == True):
        analyze_dict["scatterplot"] = df.plot.scatter(x='x', y='y')
        print('scatterplot', analyze_dict["scatterplot"])

    if (bool_dict['lineplot'] == True):
        analyze_dict["lineplot"] = df.plot.line(x='x', y='y')
        print('lineplot', analyze_dict["lineplot"])

    if (bool_dict['heatmap'] == True):
        analyze_dict["heatmap"] = df.plot.heatmap(x='x', y='y')
        print('heatmap', analyze_dict["heatmap"])

    if (bool_dict['barplot'] == True):
        analyze_dict["barplot"] = df.plot.bar(x='x', y='y')
        print('barplot', analyze_dict["barplot"])

    if (bool_dict['piechart'] == True):
        analyze_dict["piechart"] = df.plot.pie(x='x', y='y')
        print('piechart', analyze_dict["piechart"])

    if (bool_dict['violinplot'] == True):
        analyze_dict["violinplot"] = df.plot.violin(x='x', y='y')
        print('violinplot', analyze_dict["violinplot"])
    
    if (bool_dict['kdeplot'] == True):
        analyze_dict["kdeplot"] = df.plot.kde(x='x', y='y')
        print('kdeplot', analyze_dict["kdeplot"])

    if (bool_dict['hexbinplot'] == True):
        analyze_dict["hexbinplot"] = df.plot.hexbin(x='x', y='y')
        print('hexbinplot', analyze_dict["hexbinplot"])

    if (bool_dict['scatter_matrix'] == True):
        analyze_dict["scatter_matrix"] = pd.plotting.scatter_matrix(df)
        print('scatter_matrix', analyze_dict["scatter_matrix"])

    if (bool_dict['parallel_coordinates'] == True):
        analyze_dict["parallel_coordinates"] = pd.plotting.parallel_coordinates(df, 'class')
        print('parallel_coordinates', analyze_dict["parallel_coordinates"])

    if (bool_dict['andrews_curves'] == True):
        analyze_dict["andrews_curves"] = pd.plotting.andrews_curves(df, 'class')
        print('andrews_curves', analyze_dict["andrews_curves"])

    if (bool_dict['radviz'] == True):
        analyze_dict["radviz"] = pd.plotting.radviz(df, 'class')
        print('radviz', analyze_dict["radviz"])

    if (bool_dict['lag_plot'] == True):
        analyze_dict["lag_plot"] = pd.plotting.lag_plot(df)
        print('lag_plot', analyze_dict["lag_plot"])

    if (bool_dict['autocorrelation_plot'] == True):
        analyze_dict["autocorrelation_plot"] = pd.plotting.autocorrelation_plot(df)
        print('autocorrelation_plot', analyze_dict["autocorrelation_plot"])

    if (bool_dict['bootstrap_plot'] == True):
        analyze_dict["bootstrap_plot"] = pd.plotting.bootstrap_plot(df)
        print('bootstrap_plot', analyze_dict["bootstrap_plot"])

    return analyze_dict

def unique_stocks_by_date(df):
    unique_stocks_by_date = df.groupby('BarDate')['Ticker'].nunique()

    # Ensure 'BarDate' is datetime
    df['BarDate'] = pd.to_datetime(df['BarDate'])
                                        
    # select the rows of unique_stocks_by_date which have a value higher than the first value of column 'Ticker'
    unique_stocks_by_date = unique_stocks_by_date[unique_stocks_by_date > unique_stocks_by_date.iloc[0]]
    
    # Use a custom style for the plot
    plt.style.use('seaborn-darkgrid')

    # Create a line plot of unique stocks by date
    fig, ax = plt.subplots(figsize=(30, 10))
    
    # Add gridlines
    ax.grid(True)
    
    ax.plot(unique_stocks_by_date.index, unique_stocks_by_date.values)

    # Set plot title and labels
    ax.set_title('Unique Stocks by Date')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Unique Stocks')

    # Locate years and set the format to only display years
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Rotates and right aligns the x labels so they don't crowd each other
    fig.autofmt_xdate()

    # Show the plot
    plt.show()

def plot_all_stocks(df,max_AdjustedClose):
    # Use a custom style for the plot
    plt.style.use('seaborn-darkgrid')

    # Set the figure size and create the plot
    fig, ax = plt.subplots(figsize=(30, 10))

    # Add gridlines
    ax.grid(True)

    # Iterate through the groups (unique stocks)
    for stock, group in df[df["AdjustedClose"] < max_AdjustedClose].groupby('Ticker'):
        # Plot each stock's price with date on the x-axis and price on the y-axis
        ax.plot(group['BarDate'], group['AdjustedClose'], label=stock, linewidth=2, alpha=0.7)

    # Set plot title and labels with custom font sizes
    ax.set_title('AdjustedClose by Stock', fontsize=20)
    ax.set_xlabel('Date', fontsize=16)
    ax.set_ylabel('AdjustedClose', fontsize=16)

    # # legend
    # legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), framealpha=0.5)
    # for text in legend.get_texts():
    #     text.set_fontsize(12)

    # Show the plot
    plt.show()


