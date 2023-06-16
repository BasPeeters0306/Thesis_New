import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, 
                             cohen_kappa_score)
import matplotlib as plt

def prediction_metrics(df_classifications, str_model, str_single_metric):
    """
    Calculates the accuracy score for the whole series and all quantiles.
    
    Parameters
    ----------
    df_classifications : DataFrame
        Dataframe containing the classifications for the test set.

    str_model: string
        String containing the name of the model.
    
    str_single_metric: string
        String containing the name of the metric to be calculated.
        
    Returns
    -------
    classifications_metric : list
        List containing the metric score for the whole series and all quantiles.
    """
    if (str_model == "lr"):
        df_classifications["classifications"] = df_classifications["classifications_lr"]
    if (str_model == "enet"):
        df_classifications["classifications"] = df_classifications["classifications_enet"]
    if (str_model == "rf"):
        df_classifications["classifications"] = df_classifications["classifications_rf"]
    if (str_model == "gbc"):
        df_classifications["classifications"] = df_classifications["classifications_gbc"]
    if (str_model == "nn"):
        df_classifications["classifications"] = df_classifications["classifications_nn"]
    if (str_model == "lstm"):
        df_classifications["classifications"] = df_classifications["classifications_lstm"]

    # print("df_classifications: ", df_classifications)

    # Initialize array to be filled with accuracy scores for this model
    classifications_metric = []
    
    # Accuracy score for all classifications in test set

    temp = prediction_metrics_single(df_classifications)
    # print("temp: ", temp)
    temp2 = temp[str_single_metric]
    # print("temp2: ", temp2)

    classifications_metric.append(prediction_metrics_single(df_classifications)[str_single_metric])

    # print("classifications_metric: ", classifications_metric)

    # Group by date
    grouped = df_classifications.groupby("BarDate")


    # Test!!!!
    # Select top 10, reset index, calculate str_single_metric of quantile
    quantile = grouped.apply(lambda x: x.sort_values(by='classifications', 
                                                     ascending=False).iloc[: 10])
    quantile = quantile.reset_index(drop=True)
    classifications_metric.append(prediction_metrics_single(quantile)[str_single_metric])

    # Select top 20, reset index, calculate str_single_metric of quantile
    quantile = grouped.apply(lambda x: x.sort_values(by='classifications', 
                                                     ascending=False).iloc[: 20])
    quantile = quantile.reset_index(drop=True)
    classifications_metric.append(prediction_metrics_single(quantile)[str_single_metric])
    
    # Select top decile, reset index, calculate str_single_metric of quantile
    quantile = grouped.apply(lambda x: x.sort_values(by='classifications', 
                                                     ascending=False).iloc[: int(len(x)*0.1)])
    quantile = quantile.reset_index(drop=True)
    classifications_metric.append(prediction_metrics_single(quantile)[str_single_metric])

    # Select first quantile, reset index, calculate str_single_metric of quantile
    quantile = grouped.apply(lambda x: x.sort_values(by='classifications', ascending=False).iloc[ :int(len(x)*0.2)])
    quantile = quantile.reset_index(drop=True)
    classifications_metric.append(prediction_metrics_single(quantile)[str_single_metric])
    
    # Select second quantile, reset index, calculate str_single_metric of quantile
    quantile = grouped.apply(lambda x: x.sort_values(by='classifications', 
                                                     ascending=False).iloc[int(len(x)*0.2) :int(len(x)*0.4)])
    quantile = quantile.reset_index(drop=True)
    classifications_metric.append(prediction_metrics_single(quantile)[str_single_metric])
    
    # Select third quantile, reset index, calculate str_single_metric of quantile
    quantile = grouped.apply(lambda x: x.sort_values(by='classifications', 
                                                     ascending=False).iloc[int(len(x)*0.4) :int(len(x)*0.6)])
    quantile = quantile.reset_index(drop=True)
    classifications_metric.append(prediction_metrics_single(quantile)[str_single_metric])
    
    # Select fourth quantile, reset index, calculate str_single_metric of quantile
    quantile = grouped.apply(lambda x: x.sort_values(by='classifications', 
                                                     ascending=False).iloc[int(len(x)*0.6) :int(len(x)*0.8)])
    quantile = quantile.reset_index(drop=True)
    classifications_metric.append(prediction_metrics_single(quantile)[str_single_metric])
    
    # Select fifth quantile, reset index, calculate str_single_metric of quantile
    quantile = grouped.apply(lambda x: x.sort_values(by='classifications', 
                                                     ascending=False).iloc[int(len(x)*0.8) : ])
    quantile = quantile.reset_index(drop=True)
    classifications_metric.append(prediction_metrics_single(quantile)[str_single_metric])

    # Select bottom decile, reset index, calculate str_single_metric of quantile
    quantile = grouped.apply(lambda x: x.sort_values(by='classifications', 
                                                     ascending=False).iloc[int(len(x)*0.9):])
    quantile = quantile.reset_index(drop=True)
    classifications_metric.append(prediction_metrics_single(quantile)[str_single_metric])

    # Select bottom 20, reset index, calculate str_single_metric of quantile
    quantile = grouped.apply(lambda x: x.sort_values(by='classifications', 
                                                     ascending=False).iloc[int(len(x))-20: ])
    quantile = quantile.reset_index(drop=True)
    classifications_metric.append(prediction_metrics_single(quantile)[str_single_metric])

    # Select bottom 10, reset index, calculate str_single_metric of quantile
    quantile = grouped.apply(lambda x: x.sort_values(by='classifications', 
                                                     ascending=False).iloc[int(len(x))-10 : ])
    quantile = quantile.reset_index(drop=True)
    classifications_metric.append(prediction_metrics_single(quantile)[str_single_metric])

    return(classifications_metric)


def prediction_metrics_single(df_classifications_subset):  
    """
    Calculates the accuracy score for a series of classifications and a series of true values.

    Parameters
    ----------
    df_classifications_subset : DataFrame
        DataFrame containing the classifications and true values for each quantile or for the entire set
    
    Returns
    -------
    accuracy : float
        accuracy score.
    """
    # Specify y_true and y_pred
    y_true = df_classifications_subset["Target"]
    y_pred = df_classifications_subset["classifications"]

    # Calculate evaluation metrics
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "Confusion Matrix": confusion_matrix(y_true, y_pred),
        "ROC AUC Score": roc_auc_score(y_true, y_pred),
        "Cohen's Kappa": cohen_kappa_score(y_true, y_pred)
    }

    # # Print evaluation metrics
    # for metric, value in metrics.items():
    #     if metric == "Confusion Matrix":
    #         print(f"{metric}:\n{value}")
    #     else:
    #         print(f"{metric}: {value:.4f}")

    # # Calculate and plot ROC curve
    # fpr, tpr, _ = roc_curve(y_true, y_pred)
    # plt.plot(fpr, tpr)
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate (Recall)")
    # plt.title("ROC Curve")
    # plt.show()

    # # Calculate and plot Precision-Recall curve
    # precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred)
    # plt.plot(recall_curve, precision_curve)
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # plt.title("Precision-Recall Curve")
    # plt.show()

    return metrics







# Import Libraries
import statistics
import scipy.stats

# Change part of this code!!!!!!!!!!!!!! from seminar
# Only thing which needs to be changed (if we want) is that we should only consider the top and bottom predictions !!!!!!!!!!!!!!!!!!!!!!!!!
def Diebold_Mariano(df_classifications, df_classifications_lstm):  

    # To compare LSTM we must omit some of the first predictions, since they are not based on the same amount of data



    #df_DM = df_classifications[]

    # add a new column to df_DM which takes on value 0 if Target = classifications, 1 otherwise
    df_DM["DM"] = np.where(df_DM["Target"] == df_DM["classifications"], 0, 1)


    # # Specify y_true and y_pred
    # y_true = df_DM["Target"]
    # y_pred = df_DM["classifications"]


    def diebold_mariano_variable_gu(predictionErrormodel1, predictionErrormodel2):
        """
        Calculates the Diebold Mariano variable d12 at each time t=1, ..., T
        
        Parameters
        ----------
        predictionErrormodel1 : Series
            Series of prediction errors for model 1.
        predictionErrormodel2 : Series
            Series of prediction errors for model 2.
        
        Returns
        -------
        d12_t : float
            Diebold Mariano variable d12 for a given time t
        """
        # Perform a Diebold Mariano Test for each time observation across the entire cross section
        d = predictionErrormodel1 ** 2 - predictionErrormodel2 ** 2
        # Determine d12 of entire cross section at time t
        d12_t = statistics.mean(d)
        return d12_t
    

    def dmStat(d12vectorTime):
        """
        Calculates the Diebold Mariano statistic dm_stat and the corresponding p-value

        Parameters
        ----------
        d12vectorTime : Series
            Series of Diebold Mariano variables d12 for each time t=1, ..., T
        
        Returns
        -------
        dmStat : float
            Diebold Mariano statistic dm_stat
        """
        
        dBar = statistics.mean(d12vectorTime) # Check whether mean correctly obtained
        dStDev = np.std(d12vectorTime, ddof=1) / np.sqrt(np.size(d12vectorTime)) # To compute sample st dev.
        dmStat = dBar / dStDev
        pValue = scipy.stats.norm.sf(abs(dmStat))*2 # Two-sided z-test
        return dmStat
    
    # Make list of all models used
    Models = list(df_DM.columns[2:])                                                                # change this

    # Create table with prediction errors for all Models
    df_predErrorTable = pd.DataFrame()

    i = 0
    while i < len(Models):
        for model in Models:
            predErrorVector = np.where(df_DM["Target"] == df_DM["classifications"], 0, 1)                        
            df_predErrorTable[model] = predErrorVector
            i = i + 1

    df_predErrorTable = df_predErrorTable.assign(Date = df_DM.index )
    # Create an empty (zeros) Matrix which will collects p-values of DM-Stats across all Models
    numberColumns = len(Models)
    dmTable = np.zeros((len(Models), len(Models)))

    # Loop over all models to calculate corresponding p-values and put them in Matrix
    for i in range(len(Models)):
        for j in range(len(Models)):
            # Code that extracts DM-Stat for comparison model 1 and model 2
            d12Vector = np.empty(shape = 0)
            for date in df_predErrorTable["Date"].unique():
                df_day = df_predErrorTable[df_predErrorTable["Date"]== date]
                predErrors1New = df_day.iloc[:,i] 
                predErrors2New = df_day.iloc[:,j]
                d12 = diebold_mariano_variable_gu(predErrors1New, predErrors2New)
                d12Vector = np.append(d12Vector, d12)
            
            dmTable[i,j] = dmStat(d12Vector)

    dmTable = pd.DataFrame(dmTable, index = Models, columns = Models)
    # dmTable.to_excel("dmTableCrisis.xlsx", sheet_name="DM Stats")


    return dmTable







