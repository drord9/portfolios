import numpy as np
import pandas as pd
from scipy.optimize import minimize

def get_max_sharp_portfolio(train_data: pd.DataFrame, tau=0.0) -> pd.Series:

    start_date = train_data.index[0]
    end_date = train_data.index[-1]
    all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')
    close = train_data['Adj Close'].reindex(all_weekdays).fillna(method='ffill').dropna(axis=0, how="all").dropna(axis=1, how="any")

    # Relative returns
    returns = close.pct_change(1)
    Rf = 0.0
    R = returns.mean() - Rf

    # Covariance
    n = R.shape[0]
    C = returns.cov()
    C_inv = pd.DataFrame(np.linalg.inv(C.values), columns=C.columns, index=C.index)
    e = np.ones(n)

    if tau > 0.0:
        def calc_sharp(w): return (w.T @ R) / np.sqrt(w.T @ C @ w)
        def target(w): return -1 * calc_sharp(w) + tau * np.linalg.norm(w, 1)

        # create constraint optimization
        cons = ({'type': 'eq', 'fun': (lambda w: w.sum()-1)})
        bounds = None
        init_guess = [1/n] * n
        X = minimize(target, init_guess, method='SLSQP', bounds=bounds, constraints=cons).x
    else:
        X = (C_inv @ R) / (e.T @ C_inv @ R)
    
    X = pd.Series(data=X, index=R.index)

    # The portfolio we calculated is missing the simbols that has no history data (we used `dropna`)
    # so we restore them with zero weight
    return pd.Series(data=X, index=train_data['Adj Close'].columns).fillna(0)


def get_min_var_portfolio(train_data: pd.DataFrame, tau=0.0) -> pd.Series:

    start_date = train_data.index[0]
    end_date = train_data.index[-1]
    all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')
    close = train_data['Adj Close'].reindex(all_weekdays).fillna(method='ffill').dropna(axis=0, how="all").dropna(axis=1, how="any")

    # Relative returns
    returns = close.pct_change(1)
    R = returns.mean()

    # Covariance
    n = R.shape[0]
    C = returns.cov()
    C_inv = pd.DataFrame(np.linalg.inv(C.values), columns=C.columns, index=C.index)
    e = np.ones(n)

    if tau > 0.0:
        def calc_cov(w): return w.T @ C @ w
        def target(w): return calc_cov(w) + tau * np.linalg.norm(w, 1)

        # create constraint optimization
        cons = ({'type': 'eq', 'fun': (lambda w: w.sum()-1)})
        bounds = None
        init_guess = [1 / n] * n
        X = minimize(target, init_guess, method='SLSQP', bounds=bounds, constraints=cons).x
    else:
        X = (C_inv @ e) / (e.T @ C_inv @ e)

    X = pd.Series(data=X, index=R.index)

    # The portfolio we calculated is missing the simbols that has no history data (we used `dropna`)
    # so we restore them with zero weight
    return pd.Series(data=X, index=train_data['Adj Close'].columns).fillna(0)

class Portfolio:

    def __init__(self, eps=1, c=0):
        """
        The class should load the model weights, and prepare anything needed for the testing of the model.
        The training of the model should be done before submission, here it should only be loaded
        """
        self.X = None
        self.eps = eps
        self.c = c

        self.updates = 0
        self.periods = 0

    def train(self, train_data: pd.DataFrame, method=None, history=None):
        """
        :param: train_data: a dataframe as downloaded from yahoo finance, containing about
        5 years of history, with all the training data. The following day (the first that does not
        appear in the index) is the test day.
        :return (optional): weights vector.
        """
        
        # method : 'train_minVar', 'train_maxShp', 'train_equal'

        if history is not None and history < train_data.shape[0]:
            train_data = train_data.iloc[-history:]
        
        if (method is None) or (method == 'train_equal'):
            train_data_close = train_data['Adj Close']
            X = np.ones(len(train_data_close.columns)) / len(train_data_close.columns)
            self.X = pd.Series(data=X, index=train_data_close.columns)
        elif method == 'train_minVar':
            self.X = get_min_var_portfolio(train_data)
        elif method == 'train_maxShp':
            self.X = get_max_sharp_portfolio(train_data)
        else:
            raise "Not implemanted !!!"
        
        self.X.to_pickle('../portfolio.pkl')
        return self.X.to_numpy()

    def calc_PAMR(self, data: pd.DataFrame):

        close = data['Adj Close'].fillna(method='ffill')

        # Receive stock price relative (3)
        R = close.iloc[-2:].pct_change(1).iloc[-1].fillna(0) + 1
        R_market = R.mean()
        n = len(R)

        # Calculate loss (4)
        loss = np.maximum(0, self.X @ R - self.eps)

        # Set PAMR parameter (5)
        if self.c > 0:
            tau = loss / (np.sum((R - R_market)**2) + 0.5/self.c)
        else:                
            tau = loss / (np.sum((R - R_market)**2))
        
        # Update portfolio (6)
        X_new = self.X - tau*(R - R_market)

        # Normalize portfolio (7)
        # create constraint LSE
        def lse(x): return np.sum((x - X_new)**2)
        cons = ({'type': 'eq', 'fun': (lambda x: x.sum()-1)})
        bounds = [(0, 1)] * n

        X_new = minimize(lse, x0=self.X, bounds=bounds, constraints=cons).x
        self.X = pd.Series(data=X_new, index=R.index)

    def get_portfolio(self, train_data: pd.DataFrame) -> np.ndarray:
        """
        The function used to get the model's portfolio for the next day
        :param train_data: a dataframe as downloaded from yahoo finance, containing about 5 years of history,
        with all the training data. The following day (the first that does not appear in the index) is the test day
        :return: a numpy array of shape num_stocks with the portfolio for the test day
        """

        self.calc_PAMR(train_data)

        return self.X.to_numpy()
