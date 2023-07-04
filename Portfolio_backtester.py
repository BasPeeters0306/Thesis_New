import pandas as pd
import numpy as np
import statsmodels.api as sm
import math 
from scipy import stats

def backtest(df_predictions,n_long,n_short, backtest_model, weighting_method, df_returns_portfolio, 
             returnsMatrix, long_only, short_only, b_tradingcosts): 

    # Specify some naming based on model used
    predictions = "predictions_" + backtest_model
    str_returns_portfolio = "returns_" + backtest_model + "_portfolio"
    str_returns_portfolio_cum = "returns_" + backtest_model + "_portfolio_cum"
    str_logreturns_portfolio_cum = "logreturns_" + backtest_model + "_portfolio_cum"

    # Initialize column for portfolio returns
    df_returns_portfolio[str_returns_portfolio] = 0
    df_returns_portfolio[str_returns_portfolio_cum] = 0
    
    # Initialize weight matrix
    weightMatrix = pd.DataFrame(np.zeros_like(returnsMatrix), index=returnsMatrix.index, 
                                columns=returnsMatrix.columns)

    for day in df_predictions["BarDate"].unique():
        
        # Create df per day
        pts_day = df_predictions[df_predictions["BarDate"] == day]

        # Sort based on predictions
        pts_day = pts_day.sort_values(predictions, ascending=False)

        # Select top and bottom positions
        pts_day_top = pts_day.head(n_long)
        pts_day_bottom = pts_day.tail(n_short)

        pts_day_top["Weight"] = 0
        pts_day_bottom["Weight"] = 0
        
        # Weights 
        if(weighting_method == "Equal"):
            
            # Weights for long and short positions
            pts_day_top["Weight"] = 1/(2*n_long)             
            if not n_short == 0:
                pts_day_bottom["Weight"] = -1/(2*n_short)

        elif(weighting_method == "RankBased"):
            # Weights for long positions
            pts_day['Rank'] = pts_day[predictions].rank(method='first', ascending=True)
            pts_day_portfolio = pts_day[(pts_day['Rank'] > (len(pts_day) - n_long)) | (pts_day['Rank'] <= n_short)]
            pts_day_portfolio["Weight"] = (pts_day['Rank'] - pts_day['Rank'].sum()/len(pts_day))
            pts_day_portfolio["Weight"] = pts_day_portfolio["Weight"] / pts_day_portfolio["Weight"][pts_day_portfolio["Weight"]>0].sum()
            pts_day_top = pts_day_portfolio[pts_day_portfolio["Weight"]>0]
            pts_day_bottom = pts_day_portfolio[pts_day_portfolio["Weight"]<0]
        
        # If we take only long or short side then we let weights on the other side be 0
        if (long_only == True):
            pts_day_bottom["Weight"] = 0
        if (short_only == True):
            pts_day_top["Weight"] = 0

        # Fill weight matrix
        weightMatrix.loc[day, pts_day_top["Ticker"]] = pts_day_top[["Ticker", "Weight"]].set_index('Ticker').squeeze()
        weightMatrix.loc[day, pts_day_bottom["Ticker"]] = pts_day_bottom[["Ticker", "Weight"]].set_index('Ticker').squeeze()
        
        # Calculates actual returns of the weighted chosen long and short positions
        returns_long_actual = pts_day_top["NextdayReturn"]*pts_day_top["Weight"]
        returns_short_actual = pts_day_bottom["NextdayReturn"]*pts_day_bottom["Weight"]

        # Portfolio_returns
        returns_portfolio = returns_long_actual.sum() + returns_short_actual.sum()

        df_returns_portfolio.loc[df_returns_portfolio["BarDate"] == day, str_returns_portfolio] = returns_portfolio

    if b_tradingcosts == False: ##################################### DEZE is nu 1 naar achter geplaatst

        # Cumulative product of the returns
        df_returns_portfolio[str_returns_portfolio_cum] = (1 + df_returns_portfolio[str_returns_portfolio]).cumprod() - 1

        # Cumulative log returns
        df_returns_portfolio[str_logreturns_portfolio_cum] = np.log(1 + df_returns_portfolio[str_returns_portfolio]).cumsum()




    if b_tradingcosts == True:

  
        # Change dtypes of weightMatrix to float
        weightMatrix = weightMatrix.astype(float)
        # Calculate the daily changes in weights
        daily_changes = np.abs(weightMatrix - (weightMatrix.shift(1) * (1 + returnsMatrix))) # returnsMatrix is already t+1# Sum across assets for each day  
        dailyTurnover = np.sum(daily_changes, axis=1)
        # Drop first row of dailyTurnover
        dailyTurnover = dailyTurnover[1:]
        # Add a new first row to dailyTurnover with value 1
        dailyTurnover = np.insert(dailyTurnover, 0, 1)
        df_returns_portfolio[str_returns_portfolio] = df_returns_portfolio[str_returns_portfolio] - (0.0005 * dailyTurnover)       
        # Cumulative product of the returns
        df_returns_portfolio[str_returns_portfolio_cum] = (1 + df_returns_portfolio[str_returns_portfolio]).cumprod() - 1
        # Cumulative log returns
        df_returns_portfolio[str_logreturns_portfolio_cum] = np.log(1 + df_returns_portfolio[str_returns_portfolio]).cumsum()

    return(df_predictions, df_returns_portfolio, str_returns_portfolio_cum, str_logreturns_portfolio_cum, weightMatrix)

def metrics(df_returns_portfolio, backtest_model, df_metrics, df_stockindex_returns, returnsMatrix, weightMatrix, rf_rate_and_factors): 

    # Specify some naming based on model used
    str_returns_portfolio = "returns_" + backtest_model + "_portfolio"
    portfolio_returns = df_returns_portfolio[str_returns_portfolio]

    # Select all rows from rf_rate_and_factors where BarDate is in df_returns_portfolio
    rf_rate_and_factors = rf_rate_and_factors[rf_rate_and_factors['BarDate'].isin(df_returns_portfolio['BarDate'])]

    # total_return = portfolio_returns.sum() # returns are not additive so we cannot sum them
    mean_return = portfolio_returns.mean()
    std_dev = portfolio_returns.std()

    # calculate two-tailed p-value
    t_stat = mean_return / (std_dev / np.sqrt(len(portfolio_returns)))
    p_value = stats.t.sf(np.abs(t_stat), len(portfolio_returns)-1)*2

    # Information ratio
    # information_ratio = np.sqrt(252) * ((portfolio_returns.mean()) / std_dev)
    # information_ratio = None
    # Sharpe ratio
    sharpe_ratio = np.sqrt(252) * (((portfolio_returns - rf_rate_and_factors["RF"]).mean())  / std_dev)

    # if df_stockindex_returns == None:
    #     appraisal_ratio = None
    # else:
    # `   # Initialize lin reg model
    #     X_OLS = sm.add_constant(df_stockindex_returns["Return"])
    #     OLS_model = sm.OLS(portfolio_returns, X_OLS).fit()
    #     appraisal_ratio = OLS_model.tvalues[0]  # Obtain the t-statistic of the intercept`

    # Max Drawdown
    cum_returns = (1 + portfolio_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    max_drawdown = drawdown.min()

    # Maximum 1 day loss, equal to minimum returns
    max_1_day_loss = portfolio_returns.min()

    # # Maximum 1 year loss, 52 days in a year
    # max_1_year_loss = portfolio_returns.rolling(window=52).min().min() # Always equal to max 1 day loss

    ## Calculates the average daily turnover

    # # Change dtypes of weightMatrix to float
    # weightMatrix = weightMatrix.astype(float)
    # # Calculate the daily changes in weights
    # daily_changes = np.abs(weightMatrix - (weightMatrix.shift(1) * (1 + returnsMatrix))) # returnsMatrix is already t+1# Sum across assets for each day  
    # dailyTurnover = np.sum(daily_changes, axis=1)
    # # Drop first row of dailyTurnover
    # dailyTurnover = dailyTurnover[1:]
    # # Add a new first row to dailyTurnover with value 0, should be 1
    # dailyTurnover = np.insert(dailyTurnover, 0, 0)
    # # Average daily turnover
    # averagedailyTurnover = np.mean(dailyTurnover)
    averagedailyTurnover = None

    df_metrics[backtest_model] = [mean_return, std_dev, t_stat, p_value, sharpe_ratio, 
                                  max_drawdown, max_1_day_loss, averagedailyTurnover]
    return df_metrics



def regression(df_returns_portfolio_Equal_10_long_short, df_returns_portfolio_Equal_10_long_short_SESI, rf_rate_and_factors, str_returns_reg_models):

    df_returns_portfolio_Equal_10_long_short = df_returns_portfolio_Equal_10_long_short[['BarDate', 'returns_lr_portfolio', 'returns_rf_portfolio', 'returns_gbc_portfolio', 'returns_rnn_portfolio', 'returns_lstm_portfolio']]	
    df_returns_portfolio_Equal_10_long_short_SESI = df_returns_portfolio_Equal_10_long_short_SESI[['BarDate', 'returns_lr_portfolio', 'returns_rf_portfolio', 'returns_gbc_portfolio', 'returns_rnn_portfolio', 'returns_lstm_portfolio']]
    # Rename columns 'returns_lr_portfolio', 'returns_rf_portfolio', 'returns_gbc_portfolio', 'returns_rnn_portfolio', 'returns_lstm_portfolio' of df_returns_portfolio_Equal_10_long_short_SESI to 'returns_lr_portfolio_SESI', 'returns_rf_portfolio_SESI', 'returns_gbc_portfolio_SESI', 'returns_rnn_portfolio_SESI', 'returns_lstm_portfolio_SESI'
    df_returns_portfolio_Equal_10_long_short_SESI = df_returns_portfolio_Equal_10_long_short_SESI.rename(columns={'returns_lr_portfolio': 'returns_lr_portfolio_SESI', 'returns_rf_portfolio': 'returns_rf_portfolio_SESI', 'returns_gbc_portfolio': 'returns_gbc_portfolio_SESI', 'returns_rnn_portfolio': 'returns_rnn_portfolio_SESI', 'returns_lstm_portfolio': 'returns_lstm_portfolio_SESI'})
    # # Join the columns classifications_lr, classifications_rf and classifications_gb from df_classifications to df_classifications_10 by BarDate and Ticker
    df_portfolio_RFactors = df_returns_portfolio_Equal_10_long_short.merge(df_returns_portfolio_Equal_10_long_short_SESI[["BarDate", 'returns_lr_portfolio_SESI', 'returns_rf_portfolio_SESI', 'returns_gbc_portfolio_SESI', 'returns_rnn_portfolio_SESI', 'returns_lstm_portfolio_SESI']], how = "left", on = ["BarDate"])
    df_portfolio_RFactors = df_portfolio_RFactors.merge(rf_rate_and_factors[["BarDate", 'Mkt_min_RF', 'SMB', 'HML', 'ST_Rev', 'LT_Rev', 'Mom']], how = "left", on = ["BarDate"])
    df_portfolio_RFactors.drop(columns=['returns_lr_portfolio', "returns_rf_portfolio", "returns_gbc_portfolio", "returns_rnn_portfolio"], inplace=True)

    dict_regression_results = {}
    for str_returns_reg_model in str_returns_reg_models:
        # Regress returns_lstm_portfolio on Mkt_min_RF, SMB, HML, RF, ST_Rev, LT_Rev, Mom and an intercept
        X = df_portfolio_RFactors[['Mkt_min_RF', 'SMB', 'HML', 'ST_Rev', 'LT_Rev', 'Mom']]
        y = df_portfolio_RFactors[str_returns_reg_model]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        predictions = model.predict(X) # make the predictions by the model
        # Print out the statistics
        print(model.summary())

        # Get coefficients of the regression model
        params = model.params

        # Get the p-values of the coefficients
        p_values = model.pvalues
        # Get the standard errors of the coefficients
        std_err = model.bse

        # Get the R-squared value and the adjusted R-squared value
        r2 = model.rsquared
        r2_adj = model.rsquared_adj

        # Get the number of observations
        n_obs = model.nobs

        # Get the rmse
        rmse = np.sqrt(model.mse_resid)

        # Create a series with params, p_values, r2, r2_adj, n_obs and rmse
        df_regression_results = pd.concat([params, p_values, std_err], axis=1)
        df_regression_results.columns = ["params", "p_values", "std_err"]
        df_regression_results.loc["r2"] = r2
        df_regression_results.loc["r2_adj"] = r2_adj
        df_regression_results.loc["n_obs"] = n_obs
        df_regression_results.loc["rmse"] = rmse

        dict_regression_results[str_returns_reg_model] = df_regression_results

    return dict_regression_results