import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, 
                             cohen_kappa_score)
import matplotlib as plt
import statistics
import scipy.stats
from tqdm import tqdm

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

    classifications_metric.append(prediction_metrics_single(df_classifications)[str_single_metric])


    # Group by date
    grouped = df_classifications.groupby("BarDate")

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


def Diebold_Mariano(df_classifications, df_predictions, str_dm_type, str_sample):  


    df_classifications_predictions = df_classifications.copy()
    # left join columns predictions_lr, predictions_rf, predictions_gbc, predictions_lstm from df_predictions to df_classifications_temp by BarDate and Ticker
    df_classifications_predictions = df_classifications_predictions.join(df_predictions.loc[:, df_predictions.columns.isin(["BarDate", "Ticker", "predictions_lr", "predictions_rf", 
                                                                                                                            "predictions_gbc", "predictions_lstm"])].set_index(["BarDate", "Ticker"]), on=["BarDate", "Ticker"])

    Models = ["lr", "rf", "gbc", "lstm"]
    dict_quantiles = {}
    for model in Models:

        # Group by date
        grouped = df_classifications_predictions[["BarDate", "Ticker", "Target", f"classifications_{model}", f"predictions_{model}"]].groupby("BarDate")

        # Select top 20, reset index, calculate str_single_metric of quantile
        quantile_top = grouped.apply(lambda x: x.sort_values(by=f"predictions_{model}", 
                                                            ascending=False).iloc[: 20])
        quantile_top = quantile_top.reset_index(drop=True)
        # Select top 20, reset index, calculate str_single_metric of quantile
        quantile_bottom = grouped.apply(lambda x: x.sort_values(by=f"predictions_{model}", 
                                                            ascending=False).iloc[int(len(x))-20 : ])
        quantile_bottom = quantile_bottom.reset_index(drop=True)
        quantile = pd.concat([quantile_top, quantile_bottom])
        dict_quantiles[f"quantile_{model}"] = quantile

        # Count the number of 0's and 1's for column classifications_lr in quantile
        print(dict_quantiles[f"quantile_{model}"][f"classifications_{model}"].value_counts())

        dict_quantiles[f"quantile_{model}"][f"error_{model}"] = np.where(dict_quantiles[f"quantile_{model}"]["Target"] == dict_quantiles[f"quantile_{model}"][f"classifications_{model}"], 0, 1)

    print("The number of unique dates per model are equal")
    for model in Models:
        print(dict_quantiles[f"quantile_{model}"]["BarDate"].nunique())

    # Create df_errors
    df_errors = df_classifications[['BarDate', 'Ticker']]
    df_errors["lr"] = np.where(df_classifications["Target"] == df_classifications["classifications_lr"], 0, 1) 
    df_errors["rf"] = np.where(df_classifications["Target"] == df_classifications["classifications_rf"], 0, 1) 
    df_errors["gbc"] = np.where(df_classifications["Target"] == df_classifications["classifications_gbc"], 0, 1) 
    df_errors["lstm"] = np.where(df_classifications["Target"] == df_classifications["classifications_lstm"], 0, 1) 

    def diebold_mariano(predictionErrormodel1, predictionErrormodel2):
        # Perform a Diebold Mariano Test for each time observation across the entire cross section
        d = predictionErrormodel1 ** 2 - predictionErrormodel2 ** 2
        dBar = statistics.mean(d) # Check whether mean correctly obtained
        dStDev = np.std(d, ddof=1) / np.sqrt(np.size(d)) # To compute sample st dev.
        dmStat = dBar / dStDev
        pValue = scipy.stats.norm.sf(abs(dmStat))*2 # Two-sided z-test
        return dmStat, pValue

    def diebold_mariano_variable_gu(predictionErrormodel1, predictionErrormodel2):
        # Perform a Diebold Mariano Test for each time observation across the entire cross section
        d = predictionErrormodel1 ** 2 - predictionErrormodel2 ** 2
        # Determine d12 of entire cross section at time t
        d12_t = statistics.mean(d)
        return d12_t

    def dmStat(d12vectorTime):
        dBar = statistics.mean(d12vectorTime) # Check whether mean correctly obtained
        dStDev = np.std(d12vectorTime, ddof=1) / np.sqrt(np.size(d12vectorTime)) # To compute sample st dev.
        dmStat = dBar / dStDev
        pValue = scipy.stats.norm.sf(abs(dmStat))*2 # Two-sided z-test
        return dmStat, pValue

    # Make list of all models used
    Models = ["lr", "rf", "gbc", "lstm"]                                                              

    # Create an empty (zeros) Matrix which will collect p-values of DM-Stats across all Models
    dmTable = np.zeros((len(Models), len(Models)))
    pValueTable = np.zeros((len(Models), len(Models)))

    # Loop over all models to calculate corresponding p-values and put them in Matrix
    for i, model_1 in tqdm(enumerate(Models)):
        for j, model_2 in enumerate(Models):
            # Code that extracts DM-Stat for comparison model 1 and model 2
            d12Vector = np.empty(shape = 0)
                
            if (str_sample == "full"):   
                if (str_dm_type == "classic"):	
                    predErrors1New = df_errors[model_1] 
                    print("predErrors1New: ", predErrors1New)
                    predErrors2New = df_errors[model_2]
                    print("predErrors2New: ", predErrors2New)
                    d12 = diebold_mariano(predErrors1New, predErrors2New)
                    print("d12: ", d12)
                    d12Vector = np.append(d12Vector, d12)
                    print("d12Vector: ", d12Vector)

                    dmTable[i,j], pValueTable[i,j] = dmStat(d12Vector)

                elif (str_dm_type == "adjusted"):
                    for date in df_errors["BarDate"].unique():
                        df_day = df_errors[df_errors["BarDate"]== date]
                        predErrors1New = df_day[model_1] 
                        predErrors2New = df_day[model_2]
                        d12 = diebold_mariano_variable_gu(predErrors1New, predErrors2New)
                        d12Vector = np.append(d12Vector, d12)
                    
                    dmTable[i,j], pValueTable[i,j] = dmStat(d12Vector)

            if (str_sample == "Top_bottom_20"):  
                if (str_dm_type == "classic"):	
                    print("ERROR: ..........................Applying the top bottom quintile with the classic method is not possible...............................")
                    break

                elif (str_dm_type == "adjusted"):
                    for date in dict_quantiles[f"quantile_{model_1}"]["BarDate"].unique():                                          
                        df_day_model1 = dict_quantiles[f"quantile_{model_1}"][dict_quantiles[f"quantile_{model_1}"]["BarDate"]== date]
                        df_day_model2 = dict_quantiles[f"quantile_{model_2}"][dict_quantiles[f"quantile_{model_2}"]["BarDate"]== date]
                        predErrors1New = df_day_model1[f"error_{model_1}"] 
                        predErrors2New = df_day_model2[f"error_{model_2}"]
                        d12 = diebold_mariano_variable_gu(predErrors1New, predErrors2New)
                        d12Vector = np.append(d12Vector, d12)
                    
                    dmTable[i,j], pValueTable[i,j] = dmStat(d12Vector)

    dmTable = pd.DataFrame(dmTable, index = Models, columns = Models)
    pValueTable = pd.DataFrame(pValueTable, index = Models, columns = Models)

    return dmTable, pValueTable







