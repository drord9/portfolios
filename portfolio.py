import numpy as np
import pandas as pd
import cvxpy as cp
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

    if tau > 0.0:
        def calc_sharp(w): return (w.T @ R) / np.sqrt(w.T @ C @ w)
        def target(w): return -1 * calc_sharp(w) + tau * np.linalg.norm(w, 1)

        # create constraint optimization
        cons = ({'type': 'eq', 'fun': (lambda w: w.sum()-1)})
        bounds = None
        init_guess = [1/n] * n
        X = minimize(target, init_guess, method='SLSQP', bounds=bounds, constraints=cons).x
    else:
        C_inv = pd.DataFrame(np.linalg.inv(C.values), columns=C.columns, index=C.index)
        e = np.ones(n)
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

    if tau > 0.0:
        x = cp.Variable(n)
        objective = cp.Minimize((1 / 2) * cp.quad_form(x, C) + tau * cp.norm(x, 1))
        constraints = [cp.sum(x) == 1]
        result = cp.Problem(objective, constraints).solve()
        X = x.value
    else:
        C_inv = pd.DataFrame(np.linalg.inv(C.values), columns=C.columns, index=C.index)
        e = np.ones(n)
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
        
        try:
            self.X = pd.read_pickle('portfolio.pkl')
        except:
            self.X = None
            print("Can't open the weights file")
        
        self.eps = eps
        self.c = c

    def train(self, train_data: pd.DataFrame, history=300, method='train_maxShp', tau=0.0):
        """
        train_data: A dataframe as downloaded from yahoo finance, containing about 5 years of history.
        history: The number of history periods that will be used for training (None - use all history)
        method: The training method: 'train_minVar', 'train_maxShp', 'train_equal'
        tau: The weight of the regularization term
        
        :return (optional): weights vector.
        """

        if history is not None and history < train_data.shape[0]:
            train_data = train_data.iloc[-history:]
        
        if method == 'train_equal':
            train_data_close = train_data['Adj Close']
            X = np.ones(len(train_data_close.columns)) / len(train_data_close.columns)
            self.X = pd.Series(data=X, index=train_data_close.columns)
        elif method == 'train_minVar':
            self.X = get_min_var_portfolio(train_data, tau=tau)
        elif method == 'train_maxShp':
            self.X = get_max_sharp_portfolio(train_data, tau=tau)
        else:
            raise "Not implemanted !!!"
        
        self.X.to_pickle('portfolio.pkl')
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
        X = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(X - X_new))
        constraints = [cp.sum(X) == 1, X >= 0]
        result = cp.Problem(objective, constraints).solve()
        X_new = X.value
        self.X = pd.Series(data=X_new, index=R.index)

    def get_portfolio(self, train_data: pd.DataFrame) -> np.ndarray:
        """
        The function used to get the model's portfolio for the next day
        :param train_data: a dataframe as downloaded from yahoo finance, containing about 5 years of history,
        with all the training data. The following day (the first that does not appear in the index) is the test day
        :return: a numpy array of shape num_stocks with the portfolio for the test day
                
        """
        
        if self.X is None:            
            # In case the portfolio wasn't initialized correctly
            # The fallback is to initialize the weights vector equally.
            print("The model wasn't trained! Initialize the weights vector equally.")
            self.train(train_data, method='train_equal')     
            
        self.calc_PAMR(train_data)

        return self.X.to_numpy()
