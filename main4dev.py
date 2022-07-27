import pandas as pd
import yfinance as yf
from portfolio import Portfolio
import numpy as np
import matplotlib.pyplot as plt


#START_DATE = '2017-08-01'
#END_TRAIN_DATE = '2022-06-30'
#END_TEST_DATE = '2022-07-26'


def get_data(start_date, end_train_date, end_test_date):

    try:
        # try to load the data from file
        data = pd.read_pickle('data.pkl')
    except:
        # download data and save to file, so we don't need to download it again
        wiki_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp_tickers = wiki_table[0]
        tickers = [ticker.replace('.', '-') for ticker in sp_tickers['Symbol'].to_list()]
        data = yf.download(tickers, start_date, end_test_date)
        data.to_pickle('data.pkl')
    return data


def get_minVariancePortfolio(data: pd.DataFrame):

    close = data['Adj Close'].fillna(method='ffill').dropna(axis=1, how="all")

    # Relative returns
    returns = close.pct_change(1)

    # Covariance
    C = returns.cov()
    C_inv = pd.DataFrame(np.linalg.inv(C.values), columns=C.columns, index=C.index)
    e = np.ones(C_inv.shape[0])

    # Find the minimum variance portfolio
    X_min_var = (C_inv @ e) / (e.T @ C_inv @ e)
    X_min_var = pd.Series(data=X_min_var, index=data['Adj Close'].columns).fillna(0)
    return X_min_var.to_numpy()


def test_portfolio(start_date, end_train_date, end_test_date):
    full_train = get_data(start_date, end_train_date, end_test_date)

    returns = []
    log_returns = []
    strategy = Portfolio()

    ###
    train_dates = pd.date_range(start=start_date, end=end_train_date, freq='B')
    #full_train = full_train.fillna(method='ffill').dropna(axis=1, how="all")
    #full_train = full_train.fillna(method='ffill')
    train_data = full_train.reindex(train_dates)
    p_strategy = strategy.train(train_data)

    if False:
        for i, t in enumerate(train_data['Volume'].columns):
            print(i, " ", t, ": ", p_strategy[i])

    p_market = full_train['Volume'].fillna(0).div(full_train['Volume'].sum(axis=1), axis=0)
    p_minVar = get_minVariancePortfolio(train_data)
    ###

    for test_date in pd.date_range(end_train_date, end_test_date):
        if test_date not in full_train.index:
            continue
        train = full_train[full_train.index < test_date]
        cur_portfolio = strategy.get_portfolio(train)
        if not np.isclose(cur_portfolio.sum(), 1):
            raise ValueError(f'The sum of the portfolio should be 1, not {cur_portfolio.sum()}')
        test_data = full_train['Adj Close'].loc[test_date].to_numpy()
        prev_test_data = train['Adj Close'].iloc[-1].to_numpy()
        test_data = test_data / prev_test_data - 1
        test_data = np.nan_to_num(test_data)
        cur_return = cur_portfolio @ test_data

        #######
        p_market_return = p_market.loc[test_date].to_numpy() @ test_data
        p_minVar_return = p_minVar @ test_data

        log_test_data = np.log(test_data + 1)
        log_cur_return = cur_portfolio @ log_test_data
        log_p_market_return = p_market.loc[test_date].to_numpy() @ log_test_data
        log_p_minVar_return = p_minVar @ log_test_data
        #######

        returns.append({'date': test_date, 'return': cur_return, 'p_market_return': p_market_return, 'p_minVar_return': p_minVar_return})
        log_returns.append({'date': test_date, 'return': log_cur_return, 'p_market_return': log_p_market_return, 'p_minVar_return': log_p_minVar_return})
    returns = pd.DataFrame(returns).set_index('date')
    mean_return, std_returns = returns.mean(), returns.std()
    sharpe = mean_return / std_returns
    print(end_train_date)
    print(sharpe)

    """
    plt.plot(100*returns['return'], label="return")
    plt.plot(100*returns['p_market_return'], label="p_market_return")
    plt.plot(100*returns['p_minVar_return'], label="p_minVar_return")
    plt.legend(loc="best", fontsize=15)
    plt.xticks(rotation=90)
    plt.ylabel('Daily Relative Returns (%)')
    plt.suptitle('train dates:' + start_date + " - " + end_train_date)
    plt.tight_layout()
    plt.show()
    """

    log_returns = pd.DataFrame(log_returns).set_index('date')

    plt.plot(100*(np.exp(log_returns['return'].cumsum()) - 1), label="return")
    plt.plot(100*(np.exp(log_returns['p_market_return'].cumsum()) - 1), label="p_market_return")
    plt.plot(100*(np.exp(log_returns['p_minVar_return'].cumsum()) - 1), label="p_minVar_return")
    plt.legend(loc="best", fontsize=15)
    plt.xticks(rotation=90)
    plt.ylabel('Total Relative Returns (%)')
    plt.suptitle('train dates:' + start_date + " - " + end_train_date)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    #2021
    _START_DATE_ = ['2015-12-01', '2016-01-01', '2016-02-01', '2016-03-01', '2016-04-01', '2016-05-01', '2016-06-01', '2016-07-01', '2016-08-01', '2016-09-01', '2016-10-01', '2016-11-01']
    _END_TRAIN_DATE_ = ['2020-12-31', '2021-01-31', '2021-02-28', '2021-03-31', '2021-04-30', '2021-05-31', '2021-06-30', '2021-07-31', '2021-08-31', '2021-09-30', '2021-10-31', '2021-11-30']
    _END_TEST_DATE_ = ['2021-01-31', '2021-02-28', '2021-03-31', '2021-04-30', '2021-05-31', '2021-06-30', '2021-07-31', '2021-08-31', '2021-09-30', '2021-10-31', '2021-11-30', '2021-12-31']


    #2022
    #_START_DATE_ = ['2016-12-01', '2017-01-01', '2017-02-01', '2017-03-01', '2017-04-01', '2017-05-01']
    #_END_TRAIN_DATE_ = ['2021-12-31', '2022-01-31', '2022-02-28', '2022-03-31', '2022-04-30', '2022-05-31']
    #_END_TEST_DATE_ = ['2022-01-31', '2022-02-28', '2022-03-31', '2022-04-30', '2022-05-31', '2022-06-30']


    for i in range(len(_START_DATE_)):
        try:
            test_portfolio(_START_DATE_[i], _END_TRAIN_DATE_[i], _END_TEST_DATE_[i])
        except:
            print("error: ", _START_DATE_[i], _END_TRAIN_DATE_[i], _END_TEST_DATE_[i])
