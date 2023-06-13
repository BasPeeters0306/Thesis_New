import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from tqdm import tqdm
from tqdm._tqdm_notebook import tqdm_notebook
tqdm_notebook.pandas()
import importlib
import Evaluation_metrics
importlib.reload(Evaluation_metrics)
import matplotlib as plt

def GBC_test(best_model, window_size, y_in_sample, X_in_sample, y_test, X_test):
    """
    Function which performs a Gradient Boosting classification and returns the predictions and the model.

    Parameters
    ----------
    best_model : sklearn.ensemble._forest.GradientBoostingClassifier
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
    model : sklearn.ensemble._forest.GradientBoostingClassifier
        Contains the model information
    """
    

    # Initialize model, choose best hyperparameters from tuned model
    model = GradientBoostingClassifier(n_estimators = int(best_model["n_estimators"].iloc[0]), 
                                       learning_rate = best_model["learning_rate"].iloc[0], 
                                       max_depth = int(best_model["max_depth"].iloc[0]), random_state = 0)             
    # Initialize arrays which will be filled with predictions and classifications
    ar_predictions = []
    ar_classifications = []

    # Create a progress bar using tqdm
    progress_bar = tqdm(total=int(np.ceil(len(y_test)/window_size)), unit='iteration')        

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
        model.fit(X_train_window, y_train_window.squeeze())

        # Get current test target window and feature window
        if (i+window_size <= len(y_test)):
            X_test_window = X_test[i:i+window_size]
            
        else:
            X_test_window = X_test[i:]

        # Make classifications and add them to the other classifications
        classifications = model.predict(X_test_window)
        # print("classifications: \n", classifications, "classifications.shape: \n", classifications.shape)
        # change the values inside array predictions from float to int
        classifications = classifications.astype(int)
        # Make predictions and add them to the other predictions
        predictions = model.predict_proba(X_test_window)
        # print("predictions: \n", predictions, "predictions.shape: \n", predictions.shape)
        # Add predictions and classifications to array     
        ar_predictions = np.concatenate((ar_predictions, predictions[:,1]), axis=0)     
        # print("ar_predictions: \n", ar_predictions, "ar_predictions.shape: \n", ar_predictions.shape) 
        ar_classifications = np.concatenate((ar_classifications, classifications), axis=0) 
        # print("ar_classifications: \n", ar_classifications, "ar_classifications.shape: \n", ar_classifications.shape)
        
        # Update the progress bar
        progress_bar.update(1)     

        # Update i
        i += window_size
    
    # Close the progress bar
    progress_bar.close()    

    return ar_classifications, ar_predictions, model

def GBC_tune(X_train, y_train, X_val, y_val, grid):
    """
    Function which performs a grid-search to find the optimal parameters for GBC.

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
    gridsearch_results = pd.DataFrame(columns= ['n_estimators', "learning_rate", 'max_depth', "Accuracy"])
    # Loop over grid
    for i in tqdm(list(grid.values())[0]):
        for j in tqdm(list(grid.values())[1]):
            for k in list(grid.values())[2]:
                model = GradientBoostingClassifier(n_estimators = i, learning_rate = j, max_depth = k, random_state=0) 
                # print("i", i, "j", j)
                # print("X_train", X_train)
                # print("y_train", y_train)
                # print("y_train.squeeze()", y_train.squeeze())
                # try catch block to avoid errors
                try:
                    model.fit(X = X_train, y = y_train.squeeze())
                except:
                   print("Error: fitting failed")
                   continue
   
                classifications = model.predict(X_val)
                df_temp = y_val
                df_temp["classifications"] = classifications
                Accuracy = Evaluation_metrics.prediction_metrics_single(df_temp)["Accuracy"]  
                gridsearch_results.loc[len(gridsearch_results)] = [i, j, k, Accuracy]                                                                                                                                                                                                                                

    print("gridsearch_results", gridsearch_results)

    print("The lowest Accuracy value is: ", gridsearch_results["Accuracy"].max())
    print("And is reached at PC: ", gridsearch_results.loc[gridsearch_results["Accuracy"] == gridsearch_results["Accuracy"].max()].index.tolist())

    # Best model
    best_model = gridsearch_results.loc[gridsearch_results["Accuracy"] == gridsearch_results["Accuracy"].max()]
    print("Best model: ", best_model)
    return(gridsearch_results, best_model)