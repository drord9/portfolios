import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize
from scipy.optimize import least_squares

def get_equal_risk_contribution(train_data: pd.DataFrame, tau=0.0) -> pd.Series:
    
    #print("start get_equal_risk_contribution. tau=",tau)
    
    close = train_data['Adj Close'].fillna(method='ffill').dropna(axis=0, how="all").dropna(axis=1, how="any")

    # TODO:
    # # Getting all weekdays between 01/01/2018 and 01/01/2022
    # all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')

    # Relative returns
    returns = close.pct_change(1)
    Rf = 0.0
    R = returns.mean() - Rf

    # Covariance
    n = R.shape[0]
    C = returns.cov()    
    
    def target(X):
        y = X * np.dot(C, X)
        fval = 0
        for i in range(n):
            for j in range(i,n):
                Xij = y[i] - y[j]
                fval = fval + Xij*Xij
        fval = 2*fval
        return fval + tau*np.linalg.norm(X, 1)
    
    # check allocation sums to 1
    def check_sum(X):
        return np.sum(X) - 1

    # create constraint variable
    cons = ({'type': 'eq', 'fun': check_sum})
    
    # create weight boundaries
    #bounds = ((0, 1),) * n
    bounds = None
    
    # initial guess
    init_guess = [1/n] * n
    
    opt_results = minimize(target, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
    
    X = opt_results.x
    
    #print("optimal result: ", target(X))
    
    X = pd.Series(data=X, index=R.index)


    # The portfolio we calculated is missing the simbols that has no history data (we used `dropna`)
    # so we restore them with zero weight
    #print("train: portfolio size = ", X.shape, "Data size = ", train_data['Adj Close'].shape)

    return pd.Series(data=X, index=train_data['Adj Close'].columns).fillna(0)


def get_max_sharp_portfolio(train_data: pd.DataFrame, tau=0.0, isVarNormalize=False) -> pd.Series:

    close = train_data['Adj Close'].fillna(method='ffill').dropna(axis=0, how="all").dropna(axis=1, how="any")

    # TODO:
    # # Getting all weekdays between 01/01/2018 and 01/01/2022
    # all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')

    # Relative returns
    returns = close.pct_change(1)
    Rf = 0.0
    R = returns.mean() - Rf

    # Covariance
    n = R.shape[0]
    C = returns.cov()
    # from sklearn.covariance import ShrunkCovariance
    # C = ShrunkCovariance().fit(returns)
    C_inv = pd.DataFrame(np.linalg.inv(C.values), columns=C.columns, index=C.index)
    e = np.ones(n)

    if tau > 0.0:

        def calc_sharp(W):
            ret = W.T @ R
            std = np.sqrt(W.T @ C @ W)
            sharp = ret / std
            return sharp

        def neg_sharpe(W): return -1 * calc_sharp(W) + tau*np.linalg.norm(W, 1)
        def check_sum(W): return np.sum(W) - 1

        # create constraint variable
        #cons = ({'type': 'eq', 'fun': check_sum}, {'type': 'ineq', 'fun': calc_sharp})
        cons = ({'type': 'eq', 'fun': check_sum})

        # create weight boundaries
        #bounds = ((0, 1),) * n
        bounds = None

        # initial guess
        init_guess = [1/n] * n

        opt_results = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)

        X = opt_results.x

        if isVarNormalize:
             # Find the minimum variance portfolio
            X_min_var = (C_inv @ e) / (e.T @ C_inv @ e)
            X = np.multiply(X,X_min_var)
            X = X / X.sum()

    else:
        X = (C_inv @ R) / (e.T @ C_inv @ R)
    
    X = pd.Series(data=X, index=R.index)

    # The portfolio we calculated is missing the simbols that has no history data (we used `dropna`)
    # so we restore them with zero weight
    #print("train: portfolio size = ", X.shape, "Data size = ", train_data['Adj Close'].shape)

    return pd.Series(data=X, index=train_data['Adj Close'].columns).fillna(0)


def get_min_var_portfolio(train_data: pd.DataFrame, tau=0.0) -> pd.Series:

    close = train_data['Adj Close'].fillna(method='ffill').dropna(axis=0, how="all").dropna(axis=1, how="any")

    # TODO:
    # # Getting all weekdays between 01/01/2018 and 01/01/2022
    # all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')

    # Relative returns
    returns = close.pct_change(1)
    R = returns.mean()

    # Covariance
    n = R.shape[0]
    C = returns.cov()
    C_inv = pd.DataFrame(np.linalg.inv(C.values), columns=C.columns, index=C.index)
    e = np.ones(n)

    if tau > 0.0 :

        X = cp.Variable(n)

        objective = cp.Minimize((1/2)*cp.quad_form(X, C) + tau*cp.norm(X, 1))
        constraints = [cp.sum(X) == 1]
        prob = cp.Problem(objective, constraints)
        
        result = prob.solve()
        X = X.value
    else:
        X = (C_inv @ e) / (e.T @ C_inv @ e)


    X = pd.Series(data=X, index=R.index)

    # The portfolio we calculated is missing the simbols that has no history data (we used `dropna`)
    # so we restore them with zero weight
    #print("train: portfolio size = ", X.shape, "Data size = ", train_data['Adj Close'].shape)

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

        self.updates=0
        self.periods=0

    def train(self, train_data: pd.DataFrame, method=None):
        """
        :param: train_data: a dataframe as downloaded from yahoo finance, containing about
        5 years of history, with all the training data. The following day (the first that does not
        appear in the index) is the test day.
        :return (optional): weights vector.
        """
        
        # method = ['train_minVar_all', 'train_minVar_100','train_maxShp_all', 'train_maxShp_100', 'train_equal']
        
        if (method is None) or (method == 'train_equal'):
            train_data_close = train_data['Adj Close']
            X = np.ones(len(train_data_close.columns)) / len(train_data_close.columns)
            self.X = pd.Series(data=X, index=train_data_close.columns)
        elif method == 'train_minVar_all':
            self.X = get_min_var_portfolio(train_data)
        elif method == 'train_minVar_100':
            self.X = get_min_var_portfolio(train_data.iloc[-100:])
        elif method == 'train_maxShp_all':
            self.X = get_max_sharp_portfolio(train_data)
        elif method == 'train_maxShp_100':
            self.X = get_max_sharp_portfolio(train_data.iloc[-100:])
        elif method == 'train_equal_risk':
            self.X = get_equal_risk_contribution(train_data)
        elif method == 'train_equal_risk_100':
            self.X = get_equal_risk_contribution(train_data.iloc[-100:])
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
        
        # limit lambda to avoid numerical problems
        tau = min(100000, tau)

        # Update portfolio (6)
        X_new = self.X - tau*(R - R_market)

        """
        # Normalize portfolio (7)
        # create constraint LSE
        def lse(x): return np.sum((x - X_new)**2)
        cons = ({'type': 'eq', 'fun': (lambda x: x.sum()-1)})
        bounds = [(0, 1)] * n

        opt_results = minimize(lse, x0=self.X, bounds=bounds, constraints=cons)
        X_new = opt_results.x
        """

        X = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(X - X_new))
        constraints = [cp.sum(X) == 1, X >= 0]
        #constraints = [cp.sum(X) == 1]
        prob = cp.Problem(objective, constraints)

        result = prob.solve()
        X_new = X.value

        if loss > 0:
            self.updates = self.updates + 1
        self.periods = self.periods + 1

        self.X = pd.Series(data=X_new, index=R.index)


    def get_portfolio(self, train_data: pd.DataFrame) -> np.ndarray:
        """
        The function used to get the model's portfolio for the next day
        :param train_data: a dataframe as downloaded from yahoo finance, containing about 5 years of history,
        with all the training data. The following day (the first that does not appear in the index) is the test day
        :return: a numpy array of shape num_stocks with the portfolio for the test day
        """
        #train_data = train_data['Adj Close']
        #return np.ones(len(train_data.columns)) / len(train_data.columns)

        self.calc_PAMR(train_data)

        return self.X.to_numpy()
        #return get_max_sharp_portfolio(train_data).to_numpy()

    
    def get_update(self):
        return (self.updates, self.periods)

