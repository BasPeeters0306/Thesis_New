import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def logistic_regression(window_size, y_in_sample, X_in_sample, y_test, X_test, b_ALE_plot):
    """
    Function which performs a logistic regression and returns the predictions and the model.

    Parameters
    ----------
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
    ar_predictions: np.array
        Array which contains the predictions of the logistic regression model.
    model_lr : sklearn.linear_model._base.LinearRegression
        Contains the model information
    """
    
    # Initialize log reg model
    model = LogisticRegression()

    # Initialize arrays which will be filled with predictions and classifications
    ar_predictions = []
    ar_classifications = []

    i = 0
    while i <= len(y_test):
        # Get current training target window and feature window
        if(b_ALE_plot == True):
            i += window_size*5
            print("i: ", i)
            print("window_size*5: ", window_size*5)
            y_train_window = pd.concat([y_in_sample, y_test[(i-window_size*5):i]])   
            X_train_window = pd.concat([X_in_sample, X_test[(i-window_size*5):i]])   
        else: 
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
        
        # Update i
        i += window_size
 
    X_train_window_final = X_train_window
    return ar_classifications, ar_predictions, model, X_train_window_final
