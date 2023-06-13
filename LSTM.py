import numpy as np
import pandas as pd

import tensorflow
# from tensorflow import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Dropout         #Input, Concatenate         
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping

from tqdm import tqdm
from tqdm._tqdm_notebook import tqdm_notebook
tqdm_notebook.pandas()
import importlib
import Evaluation_metrics
importlib.reload(Evaluation_metrics)
# import matplotlib as plt
import matplotlib.pyplot as plt


def data_preparation(X_in_sample, X_test, y_in_sample, y_test, X_train, X_val, y_train, y_val, time_steps, b_sentiment_score, n_past_returns):

    def create_lstm_data(stock_df, time_steps):
        
        seq = []
        if len(stock_df) <= time_steps:
            return seq
        for i in range(len(stock_df) - time_steps):
            v = stock_df.iloc[i:(i + time_steps)]
            seq.append(v)
        # print("seq length: ", len(seq))
        return pd.concat(seq, keys=range(len(seq)))

    if b_sentiment_score == True and n_past_returns == 1:
        X_columns = ["BarDate", "Ticker", "PreviousdayReturn", "SESI"]
    elif b_sentiment_score == False and n_past_returns == 1:
        X_columns = ["BarDate", "Ticker", "PreviousdayReturn"]
    elif b_sentiment_score == True and n_past_returns == 3:
        X_columns = ["BarDate", "Ticker", "PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3", "SESI"]
    elif b_sentiment_score == False and n_past_returns == 3:
        X_columns = ["BarDate", "Ticker", "PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3"]
 
    # if b_sentiment_score == True:
    X_in_sample_lstm_grouped = X_in_sample.groupby("Ticker")[X_columns]
    X_test_lstm_grouped = X_test.groupby("Ticker")[X_columns]
    X_train_lstm_grouped = X_train.groupby("Ticker")[X_columns]
    X_val_lstm_grouped = X_val.groupby("Ticker")[X_columns]
    # elif b_sentiment_score == False:
    #     X_in_sample_lstm_grouped = X_in_sample.groupby("Ticker")[["BarDate", "Ticker", "PreviousdayReturn"]]
    #     X_test_lstm_grouped = X_test.groupby("Ticker")[["BarDate", "Ticker", "PreviousdayReturn"]]
    #     X_train_lstm_grouped = X_train.groupby("Ticker")[["BarDate", "Ticker", "PreviousdayReturn"]]
    #     X_val_lstm_grouped = X_val.groupby("Ticker")[["BarDate", "Ticker", "PreviousdayReturn"]]

    y_in_sample_lstm_grouped = y_in_sample.groupby("Ticker")[["BarDate", "Ticker", "Target"]]
    y_test_lstm_grouped = y_test.groupby("Ticker")[["BarDate", "Ticker", "Target"]]
    y_train_lstm_grouped = y_train.groupby("Ticker")[["BarDate", "Ticker", "Target"]]
    y_val_lstm_grouped = y_val.groupby("Ticker")[["BarDate", "Ticker", "Target"]]


    X_in_sample_lstm_seq = [create_lstm_data(group, time_steps) for _, group in X_in_sample_lstm_grouped]
    print(X_in_sample_lstm_seq)
    X_in_sample_lstm_seq = [seq for seq in X_in_sample_lstm_seq if len(seq) > 0]
    X_in_sample_lstm = pd.concat(X_in_sample_lstm_seq) if X_in_sample_lstm_seq else pd.DataFrame()
    print("first is done")

    X_test_lstm_seq = [create_lstm_data(group, time_steps) for _, group in X_test_lstm_grouped]
    X_test_lstm_seq = [seq for seq in X_test_lstm_seq if len(seq) > 0]
    X_test_lstm = pd.concat(X_test_lstm_seq) if X_test_lstm_seq else pd.DataFrame()
    print("second is done")

    X_train_lstm_seq = [create_lstm_data(group, time_steps) for _, group in X_train_lstm_grouped]     
    X_train_lstm_seq = [seq for seq in X_train_lstm_seq if len(seq) > 0]
    X_train_lstm = pd.concat(X_train_lstm_seq) if X_train_lstm_seq else pd.DataFrame()
    print("third is done")

    X_val_lstm_seq = [create_lstm_data(group, time_steps) for _, group in X_val_lstm_grouped]
    X_val_lstm_seq = [seq for seq in X_val_lstm_seq if len(seq) > 0]
    X_val_lstm = pd.concat(X_val_lstm_seq) if X_val_lstm_seq else pd.DataFrame()

    y_in_sample_lstm_seq = [group[time_steps:] for _, group in y_in_sample_lstm_grouped]
    y_in_sample_lstm = pd.concat(y_in_sample_lstm_seq)
    y_test_lstm_seq = [group[time_steps:] for _, group in y_test_lstm_grouped]
    y_test_lstm = pd.concat(y_test_lstm_seq)
    y_train_lstm_seq = [group[time_steps:] for _, group in y_train_lstm_grouped]
    y_train_lstm = pd.concat(y_train_lstm_seq)
    y_val_lstm_seq = [group[time_steps:] for _, group in y_val_lstm_grouped]
    y_val_lstm = pd.concat(y_val_lstm_seq)



    def to_3d_numpy(df, time_steps, num_features):
        # Convert dataframe to numpy array
        data = df.values
        # Get the number of stocks (samples)
        num_samples = len(df) // time_steps
        # Reshape the data to 3D for LSTM (num_samples, time_steps, num_features)
        return data.reshape(num_samples, time_steps, num_features)

    # Convert and reshape
    if b_sentiment_score == True and n_past_returns == 1:
        X_in_sample_lstm = to_3d_numpy(X_in_sample_lstm[["PreviousdayReturn", "SESI"]], time_steps, 2)
        X_test_lstm = to_3d_numpy(X_test_lstm[["PreviousdayReturn", "SESI"]], time_steps, 2)
        X_train_lstm = to_3d_numpy(X_train_lstm[["PreviousdayReturn", "SESI"]], time_steps, 2)
        X_val_lstm = to_3d_numpy(X_val_lstm[["PreviousdayReturn", "SESI"]], time_steps, 2)
    elif b_sentiment_score == False and n_past_returns == 1:
        X_in_sample_lstm = to_3d_numpy(X_in_sample_lstm[["PreviousdayReturn"]], time_steps, 1)
        X_test_lstm = to_3d_numpy(X_test_lstm[["PreviousdayReturn"]], time_steps, 1)
        X_train_lstm = to_3d_numpy(X_train_lstm[["PreviousdayReturn"]], time_steps, 1)
        X_val_lstm = to_3d_numpy(X_val_lstm[["PreviousdayReturn"]], time_steps, 1)
    elif b_sentiment_score == True and n_past_returns == 3:
        X_in_sample_lstm = to_3d_numpy(X_in_sample_lstm[["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3", "SESI"]], time_steps, 4)
        X_test_lstm = to_3d_numpy(X_test_lstm[["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3", "SESI"]], time_steps, 4)
        X_train_lstm = to_3d_numpy(X_train_lstm[["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3", "SESI"]], time_steps, 4)
        X_val_lstm = to_3d_numpy(X_val_lstm[["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3", "SESI"]], time_steps, 4)
    elif b_sentiment_score == False and n_past_returns == 3:
        X_in_sample_lstm = to_3d_numpy(X_in_sample_lstm[["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3"]], time_steps, 3)
        X_test_lstm = to_3d_numpy(X_test_lstm[["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3"]], time_steps, 3)
        X_train_lstm = to_3d_numpy(X_train_lstm[["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3"]], time_steps, 3)
        X_val_lstm = to_3d_numpy(X_val_lstm[["PreviousdayReturn", "PreviousdayReturn_2", "PreviousdayReturn_3"]], time_steps, 3)

    y_in_sample_lstm = y_in_sample_lstm[["Target"]].values
    y_test_lstm = y_test_lstm[["Target"]].values
    y_train_lstm = y_train_lstm[["Target"]].values
    y_val_lstm = y_val_lstm[["Target"]].values

    return X_in_sample_lstm, X_test_lstm, y_in_sample_lstm, y_test_lstm, X_train_lstm, X_val_lstm, y_train_lstm, y_val_lstm 

def LSTM_test(best_model, window_size, y_in_sample, X_in_sample, y_test, X_test, b_sentiment_score, n_past_returns):

    # Initialize model, choose best hyperparameters from tuned model
    model = LSTM_model_1(dropout = best_model["dropout"].iloc[0], 
                                       recurrent_dropout = best_model["recurrent_dropout"].iloc[0], 
                                       learning_rate = best_model["learning_rate"].iloc[0],
                                        optimizer = best_model["optimizer"].iloc[0],
                                        sequence_length = best_model["sequence_length"].iloc[0], b_sentiment_score=b_sentiment_score, n_past_returns=n_past_returns)
    
    # Initialize array which will be filled with predictions and classifications
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
            y_train_window = np.concatenate([y_train_window, y_test[(i-window_size):i]])        #pd.concat
            X_train_window = np.concatenate([X_train_window, X_test[(i-window_size):i]])   #pd.concat
        
        # Fit model
        model.fit(X_train_window, y_train_window.squeeze(), epochs=best_model["epochs_min_val_loss"].iloc[0], batch_size=best_model["batch_size"].iloc[0], verbose = 2)      #epochs=best_model["epochs_min_val_loss"].iloc[0]

        # Get current test target window and feature window
        if (i+window_size <= len(y_test)):
            X_test_window = X_test[i:i+window_size]
            
        else:
            X_test_window = X_test[i:]

        # Make predictions and add them to the other predicitons
        predictions = model.predict(X_test_window)

        # Add predictions (percentages) to array     
        ar_predictions = np.concatenate((ar_predictions, predictions[:,0]), axis=0)  #ar_predictions = np.concatenate((ar_predictions, predictions[0]), axis=0)
        
        # Update the progress bar
        progress_bar.update(1)     

        # Update i
        i += window_size
    
    # Close the progress bar
    progress_bar.close()    

    # Classify every value in ar_predictions_lstm as 1 or 0
    ar_classifications = np.where(ar_predictions > 0.5, 1, 0)

    return ar_predictions, ar_classifications, model

def LSTM_tune(X_train, y_train, X_val, y_val, grid, b_sentiment_score, n_past_returns):
    """
    Function which performs an LSTM and returns the predictions and the model.

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature variables of the train data.
    y_train : pd.Series
        Target variable of the train data.
    X_val : pd.DataFrame
        Feature variables of the validation data.
    y_val : pd.Series
        Target variable of the validation data.
    grid : dict
        Dictionary which contains the hyperparameters to tune.

    Returns
    -------
    gridsearch_results : pd.DataFrame
        DataFrame which contains the results of the gridsearch.
    """

    # Create a dataframe to store grid-search results
    gridsearch_results = pd.DataFrame(columns=["dropout", "recurrent_dropout", "learning_rate", "batch_size", "optimizer", "sequence_length", "val_loss", "epochs_min_val_loss", "val_accuracy", "epochs_min_acc_loss", "evaluation_dict"])         

    # Loop over grid
    for dropout in tqdm(list(grid.values())[0]):
        for recurrent_dropout in tqdm(list(grid.values())[1]):
            for learning_rate in tqdm(list(grid.values())[2]):
                for batch_size in tqdm(list(grid.values())[3]):
                    for optimizer in tqdm(list(grid.values())[4]):
                        for sequence_length in tqdm(list(grid.values())[5]):
                                model = LSTM_model_1(dropout, recurrent_dropout, learning_rate, optimizer, sequence_length, b_sentiment_score, n_past_returns)

                                # Define an EarlyStopping callback
                                early_stopping = EarlyStopping(monitor='val_loss', patience=10) # this is the number of epochs with no improvement after which training will be stopped

                                # Fit the model to the data, using early stopping
                                # This is different here from the other models
                                try:
                                    history = model.fit(X_train, y_train, 
                                                        epochs=100,             # 1000 in Fischer
                                                        batch_size=batch_size,             
                                                        validation_data=(X_val, y_val), 
                                                        callbacks=[early_stopping], verbose = 2)
                                except:
                                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  Error: fitting failed using grid: dropout = ", dropout, "recurrent_dropout = ", recurrent_dropout, "learning_rate = ", learning_rate, "batch_size = ", batch_size, "optimizer = ", optimizer, "sequence_length = ", sequence_length)
                                     
                                    continue

                                evaluation_dict = history.history

                                # Add evaluation_dict to gridsearch_results with all hyperparametesr
                                gridsearch_results.loc[len(gridsearch_results)] = [dropout, recurrent_dropout, learning_rate, batch_size, optimizer, sequence_length, min(evaluation_dict["val_loss"]), 
                                                                                   evaluation_dict["val_loss"].index(min(evaluation_dict["val_loss"]))+1,  max(evaluation_dict["val_accuracy"]), evaluation_dict["val_accuracy"].index(max(evaluation_dict["val_accuracy"]))+1, 
                                                                                   evaluation_dict] 

    # Returns best model based on val_loss
    best_model = gridsearch_results.loc[gridsearch_results["val_loss"] == gridsearch_results["val_loss"].min()]
    print("Best model based on the minimum val_loss: \n", best_model.head().to_string())

    return(gridsearch_results, best_model)

def LSTM_model_1(dropout, recurrent_dropout, learning_rate, optimizer, sequence_length, b_sentiment_score, n_past_returns):

    if b_sentiment_score == True:
        n_features = n_past_returns + 1
    else:
        n_features = n_past_returns


    # Create a Sequential model
    model = Sequential()

    # Add an LSTM layer with 25 hidden neurons and input shape (sequence_length, n_features)
    # sequence_length is the number of time steps and n_features is the number of features
    model.add(LSTM(25, input_shape=(sequence_length, n_features), dropout=dropout, recurrent_dropout=recurrent_dropout))

    # Add a Dense output layer with 1 neuron and sigmoid activation
    model.add(Dense(1, activation="sigmoid"))

    # Compile the model
    if optimizer == "RMSprop":
        model.compile(loss='binary_crossentropy', 
                    optimizer=RMSprop(learning_rate=learning_rate), 
                    metrics=['accuracy'])
    elif optimizer == "adam":
        model.compile(loss='binary_crossentropy', 
                    optimizer=Adam(learning_rate=learning_rate), 
                    metrics=['accuracy'])

    return model

def plot_train_val_loss(performances_LSTM):

    # loop over all values in dataframe performances_LSTM
    for index, row in performances_LSTM.iterrows():

            # Select column with name "evaluation_dict"
            evaluation_dict = row["evaluation_dict"]       

            # plot loss
            plt.figure(figsize=(12,6))
            plt.subplot(1, 2, 1)
            loss_values = evaluation_dict['loss']
            val_loss_values = evaluation_dict['val_loss']
            epochs = range(1, len(loss_values) + 1)
            plt.plot(epochs, loss_values, 'r', label='Training loss')
            plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
            plt.title('Training and validation loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            # plot accuracy
            plt.subplot(1, 2, 2)
            acc_values = evaluation_dict['accuracy']
            val_acc_values = evaluation_dict['val_accuracy']
            plt.plot(epochs, acc_values, 'r', label='Training accuracy')
            plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
            plt.title('Training and validation accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()







# Creates LSTM model with separate time steps for both features
# def LSTM_model_2(time_steps_1, time_steps_2, X_train ):

#     # Specify first and second feature
#     X_train_feature_1 = X_train["PreviousdayReturn"]
#     X_train_feature_2 = X_train["SESI"]
#     # Define the LSTM for the first feature
#     input1 = Input(shape=(time_steps_1, 1))
#     lstm1 = LSTM(50, dropout=0.1, recurrent_dropout=0.1)(input1)

#     # Define the LSTM for the second feature
#     input2 = Input(shape=(time_steps_2, 1))
#     lstm2 = LSTM(50, dropout=0.1, recurrent_dropout=0.1)(input2)

#     # Concatenate the outputs of the two LSTMs
#     concat = concatenate([lstm1, lstm2])

#     # Add a Dense layer
#     output = Dense(1, activation='sigmoid')(concat)

#     # Create and compile the model
#     model = Model(inputs=[input1, input2], outputs=output)
#     model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

#     # Early stopping callback
#     early_stopping = EarlyStopping(monitor='val_loss', patience=10)

#     # Train the model
#     history = model.fit([X_train_feature_1, X_train_feature_2], y_train, epochs=1000, validation_data=([X_val_feature_1, X_val_feature_2], y_val), callbacks=[early_stopping], verbose = 2)

#     # Evaluate the model on the test data
#     scores = model.evaluate([X_test_feature_1, X_test_feature_2], y_test, verbose=0)
#     print("Test Accuracy: %.2f%%" % (scores[1]*100))

#     return model