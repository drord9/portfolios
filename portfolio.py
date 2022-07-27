import numpy as np
import pandas as pd


class Portfolio:

    def __init__(self):
        """
        The class should load the model weights, and prepare anything needed for the testing of the model.
        The training of the model should be done before submission, here it should only be loaded
        """
        self.X = None

    def train(self, train_data: pd.DataFrame):
        """
        :param: train_data: a dataframe as downloaded from yahoo finance, containing about
        5 years of history, with all the training data. The following day (the first that does not
        appear in the index) is the test day.
        :return (optional): weights vector.
        """

        close = train_data['Adj Close'].fillna(method='ffill').dropna(axis=1, how="all")

        # Relative returns
        returns = close.pct_change(1)
        Rf = 0.0
        R = returns.mean() - Rf

        # Covariance
        C = returns.cov()
        C_inv = pd.DataFrame(np.linalg.inv(C.values), columns=C.columns, index=C.index)
        e = np.ones(C_inv.shape[0])

        # Find the maximum sharp ratio portfolio
        X = (C_inv@R)/(e.T@C_inv@R)

        # The portfolio we calculated is missing the simbols that has no history data (we used `dropna`)
        # so we restore them with zero weight
        self.X = pd.Series(data=X ,index=train_data['Adj Close'].columns).fillna(0)

        return self.X.to_numpy()


    def get_portfolio(self, train_data: pd.DataFrame) -> np.ndarray:
        """
        The function used to get the model's portfolio for the next day
        :param train_data: a dataframe as downloaded from yahoo finance, containing about 5 years of history,
        with all the training data. The following day (the first that does not appear in the index) is the test day
        :return: a numpy array of shape num_stocks with the portfolio for the test day
        """
        #train_data = train_data['Adj Close']
        #return np.ones(len(train_data.columns)) / len(train_data.columns)

        return self.X.to_numpy()
