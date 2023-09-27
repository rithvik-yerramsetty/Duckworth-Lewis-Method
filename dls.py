import os
import pickle
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union

from collections import defaultdict as DD
from scipy.optimize import minimize
from copy import deepcopy


if not os.path.exists('../models'):
    os.makedirs('../models')
if not os.path.exists('../plots'):
    os.makedirs('../plots')


class DLModel:
    """
        Model Class to approximate the Z function as defined in the assignment.
    """

    def __init__(self):
        """Initialize the model."""
        self.Z0 = [None] * 10
        self.L = None
    
    def get_predictions(self, X, Z_0=None, w=10, L=None) -> np.ndarray:
        """Get the predictions for the given X values.

        Args:
            X (np.array): Array of overs remaining values.
            Z_0 (float, optional): Z_0 as defined in the assignment.
                                   Defaults to None.
            w (int, optional): Wickets in hand.
                               Defaults to 10.
            L (float, optional): L as defined in the assignment.
                                 Defaults to None.

        Returns:
            np.array: Predicted score possible
        """
        return Z_0*(1 - np.exp((-1*L*X)/Z_0))

    def calculate_loss(self, Params, X, Y, w=10) -> float:
        """ Calculate the loss for the given parameters and datapoints.
        Args:
            Params (list): List of parameters to be optimized.
            X (np.array): Array of overs remaining values.
            Y (np.array): Array of actual average score values.
            w (int, optional): Wickets in hand.
                               Defaults to 10.

        Returns:
            float: Mean Squared Error Loss for the model parameters 
                   over the given datapoints.
        """
        y_pred =  self.get_predictions(X=X, Z_0=Params[0], w=w, L = Params[1])
        loss = np.mean((y_pred - Y)**2)
        return loss

    
    def save(self, path):
        """Save the model to the given path.

        Args:
            path (str): Location to save the model.
        """
        with open(path, 'wb') as f:
            pickle.dump((self.L, self.Z0), f)
    
    def load(self, path):
        """Load the model from the given path.

        Args:
            path (str): Location to load the model.
        """
        with open(path, 'rb') as f:
            (self.L, self.Z0) = pickle.load(f)


def get_data(data_path) -> Union[pd.DataFrame, np.ndarray]:
    """
    Loads the data from the given path and returns a pandas dataframe.

    Args:
        path (str): Path to the data file.

    Returns:
        pd.DataFrame, np.ndarray: Data Structure containing the loaded data
    """
    df = pd.read_csv(data_path)
    return df

def correct_date(x):
    if(x.count(',') == 0):
        return x

    month_dict = {'Jan':1, 'Feb': 2, 'Mar': 3, 'Apr':4, 'May': 5, 'Jun':6, 'Jul':7, 'Aug': 8, 'Sep':9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

    mon_dates, year = x.split(',')
    mon, dates = mon_dates.split(' ')
    month = month_dict[mon]
    date,_ = dates.split('-')
    res = str(date) + '/' + str(month) + '/' + year.strip() 

    return res

def preprocess_data(data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
    """Preprocesses the dataframe by
    (i)   removing the unnecessary columns,
    (ii)  loading date in proper format DD-MM-YYYY,
    (iii) removing the rows with missing values,
    (iv)  anything else you feel is required for training your model.

    Args:
        data (pd.DataFrame, nd.ndarray): Pandas dataframe containing the loaded data

    Returns:
        pd.DataFrame, np.ndarray: Datastructure containing the cleaned data.
    """
    ##Following code is used to convert wrong date format to right dte format.
    try:
        data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, format="%d/%m/%Y").dt.strftime('%d-%m-%Y')
    except:
        data['Date'] = data['Date'].apply(correct_date)
    finally:
        data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, format="%d/%m/%Y").dt.strftime('%d-%m-%Y')

    #Following code is used to remove columns which are not necessary for training model and calculating loss(except 'Date') 
    useful_columns = ['Date', 'Innings', 'Wickets.in.Hand', 'Overs.Remaining', 'Total.Overs', 'Over', 'Runs.Remaining', 'Innings.Total.Runs', 'Runs']
    all_columns = data.columns
    df_filtered = data.drop(list(set(all_columns) - set(useful_columns)), axis=1)
    df_filtered = df_filtered[(df_filtered['Innings'] == 1) & (df_filtered['Wickets.in.Hand'] != 0)]

    ##Following code is used to add rows corresponding to u = 50 and w = 10 for each match.
    df_start = deepcopy(df_filtered.loc[df_filtered['Over'] == 1])
    df_start['Over'], df_start['Runs.Remaining'], df_start['Wickets.in.Hand'] = 0, df_start['Innings.Total.Runs'], 10
    df = pd.concat([df_start, df_filtered], axis=0)
    
    return df


def train_model(data: Union[pd.DataFrame, np.ndarray], model: DLModel) -> DLModel:
    """Trains the model

    Args:
        data (pd.DataFrame, np.ndarray): Datastructure containg the cleaned data
        model (DLModel): Model to be trained
    """

    def error(params):
        err = 0
        tot_data_points = len(data)

        for wkt_ih in range(10, 0, -1):
            data_wkt_ih = data.loc[data['Wickets.in.Hand'] == wkt_ih]
            x = data_wkt_ih['Total.Overs'] - data_wkt_ih['Over']
            y_true = data_wkt_ih['Runs.Remaining']
            y_pred = params[wkt_ih - 1]*(1 - np.exp((-1*params[10]*x)/params[wkt_ih - 1]))
            err += np.sum((y_pred - y_true)**2)/tot_data_points
        return err
    
    
    init_params = []
    for wkts_ih in range(10, 0, -1):
        init_params.append(data.loc[data['Wickets.in.Hand'] == wkts_ih]['Runs.Remaining'].mean())

    init_params.append(data.loc[(data['Innings'] == 1) & (data['Over'] == data['Total.Overs'])]['Runs'].values.mean())
    
    res = np.array(minimize(error, init_params).x)
    model.Z0 = res[:10]
    model.L = res[10]

    return model


def plot(model: DLModel, plot_path: str) -> None:
    """ Plots the model predictions against the number of overs
        remaining according to wickets in hand.

    Args:
        model (DLModel): Trained model
        plot_path (str): Path to save the plot
    """
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
    def non_linear(u, z_0):
        return z_0*(1 - np.exp((-1*model.L*u)/z_0))
    
    for wkt_ih in range(10, 0, -1):
        x = np.array(list(range(0,51)))
        plt.plot(x.reshape(-1,1), non_linear(x, model.Z0[wkt_ih-1]).reshape(-1,1), label='w = '+str(wkt_ih))
        plt.xlabel('Overs Remaining')
        plt.ylabel('Predicted Runs')
    plt.title('Predicted Runs vs Overs Remaining')
    plt.grid()
    plt.legend()
    plt.savefig(plot_path)

    del os.environ['QT_QPA_PLATFORM']


def print_model_params(model: DLModel) -> List[float]:
    '''
    Prints the 11 (Z_0(1), ..., Z_0(10), L) model parameters

    Args:
        model (DLModel): Trained model
    
    Returns:
        array: 11 model parameters (Z_0(1), ..., Z_0(10), L)

    '''
    model_params = []
    for wk_ih,z in enumerate(model.Z0):
        model_params.append(z)
        print("Z_0({0}) = {1}".format(wk_ih + 1, z))
    
    model_params.append(model.L)
    print("L = {0}".format(model.L))

    return model_params


def calculate_loss(model: DLModel, data: Union[pd.DataFrame, np.ndarray]) -> float:
    '''
    Calculates the normalised squared error loss for the given model and data

    Args:
        model (DLModel): Trained model
        data (pd.DataFrame or np.ndarray): Data to calculate the loss on
    
    Returns:
        float: Normalised squared error loss for the given model and data
    '''

    err = 0
    tot_data_points = len(data)

    for wkt_ih in range(10, 0, -1):
        data_wkt_ih = data.loc[data['Wickets.in.Hand'] == wkt_ih]
        x = data_wkt_ih['Total.Overs'] - data_wkt_ih['Over']
        y_true = data_wkt_ih['Runs.Remaining']
        params = [model.Z0[wkt_ih - 1]] + [model.L]
        err += model.calculate_loss(params,x,y_true,w=wkt_ih) * len(x)/tot_data_points
        

    print("loss= ", err)
    return err

def main(args):
    """Main Function"""

    data = get_data(args['data_path'])  # Loading the data
    print("Data loaded.")
    
    # Preprocess the data
    data = preprocess_data(data)
   
    print("Data preprocessed.")
    
    model = DLModel()  # Initializing the model
    model = train_model(data, model)  # Training the model
    model.save(args['model_path'])  # Saving the model
    
    plot(model, args['plot_path'])  # Plotting the model
    
    # Printing the model parameters
    print_model_params(model)

    # Calculate the normalised squared error
    calculate_loss(model, data)


if __name__ == '__main__':
    args = {
        "data_path": "../data/04_cricket_1999to2011.csv",
        "model_path": "../models/model.pkl",  # ensure that the path exists
        "plot_path": "../plots/plot.png",  # ensure that the path exists
    }
    main(args)
