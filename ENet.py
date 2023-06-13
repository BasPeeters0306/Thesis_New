import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from tqdm._tqdm_notebook import tqdm_notebook
tqdm_notebook.pandas()
import importlib
import Evaluation_metrics
importlib.reload(Evaluation_metrics)
import matplotlib.pyplot as plt
import warnings
# from pandas.core.common import SettingWithCopyWarning
from pandas.errors import SettingWithCopyWarning
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.utils import DataConversionWarning


# Suppress warnings
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)
warnings.simplefilter(action='ignore', category=DataConversionWarning)




def ENet_test(best_model, window_size, y_in_sample, X_in_sample, y_test, X_test):
    """
    Function which performs a logistic regression and returns the predictions and the model.

    Parameters
    ----------
    best_model : sklearn.linear_model._base.LogisticRegression
        Contains the model information
    window_size : int
        Size of the window.
    y_in_sample : pd.Series
        Target variable of the in-sample data.
    X_in_sample : pd.DataFrame
        Feature variables of the in-sample data.
    y_test : pd.Series
        Target variable of the test data.
    X_test : pd.DataFrame
        Feature variables of the test data.
    df_predictions : pd.DataFrame
        DataFrame which contains the predictions of the other models.
    df_accuracy : pd.DataFrame
        DataFrame which contains the R2 scores of the other models.

    Returns
    -------
    df_predictions : pd.DataFrame
        DataFrame which contains the predictions of the other models.
    df_accuracy : pd.DataFrame
        DataFrame which contains the R2 scores of the other models.
    model : sklearn.linear_model._base.LinearRegression
        Contains the model information
    """
    

    # Initialize model, choose best hyperparameters from tuned model
    model = LogisticRegression(penalty='elasticnet', l1_ratio=best_model["l1_ratio"].iloc[0], C=best_model["C"].iloc[0], solver='saga')         #, max_iter=10000)                  
    # Initialize arrays which will be filled with predictions and classifications
    ar_predictions = []
    ar_classifications = []

    i = 0
    while i <= len(y_test):
        # Get current training target window and feature window
        if (i==0):
            y_train_window = y_in_sample
            X_train_window = X_in_sample
        else:
            y_train_window = pd.concat([y_train_window, y_test[(i-window_size):i]])   
            X_train_window = pd.concat([X_train_window, X_test[(i-window_size):i]])  
        
        # Fit model
        model.fit(X_train_window, y_train_window)

        # Get current test target window and feature window
        if (i+window_size <= len(y_test)):
            X_test_window = X_test[i:i+window_size]
            
        else:
            X_test_window = X_test[i:]

        # Make classifications and add them to the other classifications
        classifications = model.predict(X_test_window)
        print("classifications: \n", classifications, "classifications.shape: \n", classifications.shape)
        # change the values inside array predictions from float to int
        classifications = classifications.astype(int)
        # Make predictions and add them to the other predictions
        predictions = model.predict_proba(X_test_window)
        print("predictions: \n", predictions, "predictions.shape: \n", predictions.shape)
        # Add predictions and classifications to array     
        ar_predictions = np.concatenate((ar_predictions, predictions[:,1]), axis=0)     
        print("ar_predictions: \n", ar_predictions, "ar_predictions.shape: \n", ar_predictions.shape) 
        ar_classifications = np.concatenate((ar_classifications, classifications), axis=0) 
        print("ar_classifications: \n", ar_classifications, "ar_classifications.shape: \n", ar_classifications.shape)

        # Update i
        i += window_size
      

    return ar_classifications, ar_predictions, model

def ENet_tune(X_train, y_train, X_val, y_val, grid):
    """
    Function which performs a grid-search to find the optimal parameters for ENet.

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature variables of the training data.
    y_train : pd.Series
        Target variable of the training data.
    X_val : pd.DataFrame
        Feature variables of the validation data.
    y_val : pd.Series
        Target variable of the validation data.
    grid : dict
        Dictionary which contains the parameters for the grid-search.

    Returns
    -------
    gridsearch_results : pd.DataFrame
        DataFrame which contains the performance of all models.
    best_model : pd.DataFrame
        DataFrame which contains the best model.
    """

    # Create a dataframe to store grid-search results
    gridsearch_results = pd.DataFrame(columns= ["l1_ratio", "C", "Accuracy"])
    # Loop over grid
    for i in tqdm(list(grid.values())[0]):
        for j in tqdm(list(grid.values())[1]):
                model = LogisticRegression(penalty='elasticnet', l1_ratio=i, C=j, solver='saga')                 #solver='saga', max_iter=10000)      
                model.fit(X = X_train, y = y_train)
                classifications = model.predict(X_val)
                df_temp = y_val
                df_temp["classifications"] = classifications
                Accuracy = Evaluation_metrics.prediction_metrics_single(df_temp)["Accuracy"]  
                gridsearch_results.loc[len(gridsearch_results)] = [i, j, Accuracy]                                                                                                                                                                                                                                

    print("gridsearch_results", gridsearch_results)

    # # Plots Accuracy against l1_ratio
    # plt.plot(gridsearch_results["l1_ratio"], gridsearch_results["Accuracy"])
    # plt.xlabel('alpha')
    # plt.ylabel('Accuracy')
    # plt.title('ENet results on validation set')
    # plt.show()

    # Plots Accuracy against C
    plt.plot(gridsearch_results["C"], gridsearch_results["Accuracy"])
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.title('ENet results on validation set')
    plt.show()

    print("The highest Accuracy value is: ", gridsearch_results["Accuracy"].max())
    print("And is reached at PC: ", gridsearch_results.loc[gridsearch_results["Accuracy"] == gridsearch_results["Accuracy"].max()].index.tolist())

    # Best model
    best_model = gridsearch_results.loc[gridsearch_results["Accuracy"] == gridsearch_results["Accuracy"].max()]
    print("Best model: ", best_model)
    return(gridsearch_results, best_model)