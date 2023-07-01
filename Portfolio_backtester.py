import pandas as pd
import numpy as np
import statsmodels.api as sm
import math 

def backtest(df_predictions,n_long,n_short, backtest_model, weighting_method, df_returns_portfolio, 
             returnsMatrix, long_only, short_only): 

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
            pts_day_top["Weight"] = 1/n_long
            if not n_short == 0:
                pts_day_bottom["Weight"] = -1/n_short

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

    # Information ratio
    # information_ratio = np.sqrt(252) * ((portfolio_returns.mean()) / std_dev)
    information_ratio = None
    # Sharpe ratio
    sharpe_ratio = np.sqrt(252) * (((portfolio_returns - rf_rate_and_factors["RF"]).mean())  / std_dev)

    if df_stockindex_returns == None:
        appraisal_ratio = None
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

    # Calculates the average dayly turnover
    # T = weightMatrix.shape[0]
    # totalTurnover = 0
    # for t in range(0,T-1):
    #     cumTurnover = 0
    #     for i in range(0, weightMatrix.shape[1] - 1):
    #         cumTurnoverDenom = 1
    #         for j in range(0, weightMatrix.shape[1] - 1):
    #             cumTurnoverDenom = cumTurnoverDenom + (weightMatrix.iloc[t,j] * returnsMatrix.iloc[t+1,j])
        
    #         cumTurnover = cumTurnover + abs(weightMatrix.iloc[(t + 1), i] - (weightMatrix.iloc[t, i] * 
    #                                                             (1 + returnsMatrix.iloc[(t+1),i])) / 
    #                                                             cumTurnoverDenom)
    #     totalTurnover = totalTurnover + cumTurnover
    # averagedaylyTurnover = (1/T) * totalTurnover
    averagedaylyTurnover = None

    df_metrics[backtest_model] = [mean_return, std_dev, information_ratio, sharpe_ratio, appraisal_ratio, 
                                  max_drawdown, max_1_day_loss, averagedaylyTurnover]

    return df_metrics



