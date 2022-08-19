import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize

def get_equal_risk_contribution(train_data: pd.DataFrame, tau=0.0) -> pd.Series:
    
    print("start get_equal_risk_contribution. tau=",tau)
    
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
    
    print("optimal result: ", target(X))
    
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

    def calc_sharp(W):
        ret = W.T @ R
        std = np.sqrt(W.T @ C @ W)
        sharp = ret / std
        return sharp

    def neg_sharpe(W):
        return -1 * calc_sharp(W) + tau*np.linalg.norm(W, 1)

    # check allocation sums to 1
    def check_sum(W):
        return np.sum(W) - 1


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

    X = cp.Variable(n)

    objective = cp.Minimize((1/2)*cp.quad_form(X, C) + tau*cp.norm(X, 1))
    constraints = [cp.sum(X) == 1]
    prob = cp.Problem(objective, constraints)
    
    result = prob.solve()

    X = pd.Series(data=X.value, index=R.index)

    # The portfolio we calculated is missing the simbols that has no history data (we used `dropna`)
    # so we restore them with zero weight
    #print("train: portfolio size = ", X.shape, "Data size = ", train_data['Adj Close'].shape)

    return pd.Series(data=X, index=train_data['Adj Close'].columns).fillna(0)


class Portfolio:

    def __init__(self):
        """
        The class should load the model weights, and prepare anything needed for the testing of the model.
        The training of the model should be done before submission, here it should only be loaded
        """
        self.X = None

    def train(self, train_data: pd.DataFrame, tau=0.0):
        """
        :param: train_data: a dataframe as downloaded from yahoo finance, containing about
        5 years of history, with all the training data. The following day (the first that does not
        appear in the index) is the test day.
        :return (optional): weights vector.
        """

        #self.X = get_min_var_portfolio(train_data, tau)
        #self.X = get_max_sharp_portfolio(train_data, tau)
        self.X = get_max_sharp_portfolio(train_data, tau, True)
        #self.X = get_equal_risk_contribution(train_data, tau)
        
        self.X.to_pickle('../portfolio.pkl')

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
        #return get_max_sharp_portfolio(train_data).to_numpy()

