import pandas as pd
import yfinance as yf
from portfolio import Portfolio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas_datareader.data as pd_data


def get_data(start_date, end_train_date, end_test_date):

    try:
        # try to load the data from file
        print("loading data.pkl....")
        data = pd.read_pickle('../data.pkl')
        if data.index[0] > pd.Timestamp(start_date) or data.index[-1] < pd.Timestamp(end_test_date) - pd.DateOffset(3):
            print(" data in data.pkl is stale !!! ")
            raise "GoTo except"

        mc = pd.read_pickle('../marketCap.pkl')
    except:
        # download data and save to file, so we don't need to download it again
        print("Downloading ....")
        wiki_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp_tickers = wiki_table[0]
        tickers = [ticker.replace('.', '-') for ticker in sp_tickers['Symbol'].to_list()]
        data = yf.download(tickers, start_date, end_test_date, progress=True)
        data.to_pickle('../data.pkl')

        mc = pd_data.get_quote_yahoo(tickers)['marketCap']
        mc.to_pickle('../marketCap.pkl')

    return data, mc


def get_referancePortfolios(data: pd.DataFrame, marketCap: pd.DataFrame):

    #close = data['Adj Close'].fillna(method='ffill').dropna(axis=1, how="all")
    all_data_columns = data['Adj Close'].columns
    close = data['Adj Close'].fillna(method='ffill').dropna(axis=0, how="all").dropna(axis=1, how="any")

    # Relative returns
    returns = close.pct_change(1)
    Rf = 0.0
    R = returns.mean() - Rf

    # Covariance
    C = returns.cov()
    C_inv = pd.DataFrame(np.linalg.inv(C.values), columns=C.columns, index=C.index)
    e = np.ones(C_inv.shape[0])

    # Find the minimum variance portfolio
    X_min_var = (C_inv @ e) / (e.T @ C_inv @ e)
    X_min_var = pd.Series(data=X_min_var, index=all_data_columns).fillna(0)

    # Find the maximum sharp ratio portfolio
    X_max_shp = (C_inv @ R) / (e.T @ C_inv @ R)
    X_max_shp = pd.Series(data=X_max_shp, index=all_data_columns).fillna(0)

    # Find the market portfolio
    #marketCap = marketCap.reindex(data['Adj Close'].columns)
    #X_market = marketCap.div(marketCap.sum())
    
    X_market = np.ones(len(all_data_columns)) / len(all_data_columns)

    return X_min_var.to_numpy(), X_max_shp.to_numpy(), X_market


def test_portfolio(start_date, end_train_date, end_test_date, data=None, params=(None,None,0)):

    if data is None:
        full_train, marketCap = get_data(start_date, end_train_date, end_test_date)
    else:
        full_train, marketCap = data
    
    method, history, tau = params

    returns = []
    log_returns = []
    strategy = Portfolio()

    ###
    train_dates = pd.date_range(start=start_date, end=end_train_date, freq='B')
    train_data = full_train.reindex(train_dates)
    p_strategy = strategy.train(train_data, method=method, history=history, tau=tau)

    p_minVar, p_maxShp, p_market = get_referancePortfolios(train_data, marketCap)
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
        p_market_return = p_market @ test_data
        p_minVar_return = p_minVar @ test_data
        p_maxShp_return = p_maxShp @ test_data

        log_test_data = np.log(test_data + 1)
        log_cur_return = cur_portfolio @ log_test_data
        log_p_market_return = p_market @ log_test_data
        log_p_minVar_return = p_minVar @ log_test_data
        log_p_maxShp_return = p_maxShp @ log_test_data
        #######

        returns.append({'date': test_date, 'return': cur_return, 'p_market_return': p_market_return, 'p_minVar_return': p_minVar_return, 'p_maxShp_return': p_maxShp_return})
        log_returns.append({'date': test_date, 'return': log_cur_return, 'p_market_return': log_p_market_return, 'p_minVar_return': log_p_minVar_return, 'p_maxShp_return': log_p_maxShp_return})
    returns = pd.DataFrame(returns).set_index('date')
    mean_return, std_returns = returns.mean(), returns.std()
    sharpe = mean_return / std_returns
    #print(end_train_date)
    #print(sharpe)

    show_figs = False
    if show_figs:
        fig = plt.figure()
        ax = fig.add_subplot(3, 1, 1)
        ax.plot(100*returns['return'], label="return")
        ax.plot(100*returns['p_market_return'], label="p_market_return")
        ax.plot(100*returns['p_minVar_return'], label="p_minVar_return")
        ax.plot(100 * returns['p_maxShp_return'], label="p_maxShp_return")
        ax.legend(loc="best")
        ax.tick_params(rotation=90)
        ax.set_ylabel('Daily Relative Returns (%)')

        log_returns = pd.DataFrame(log_returns).set_index('date')

        ax = fig.add_subplot(3, 1, 2)
        ax.plot(100 * (np.exp(log_returns['return'].cumsum()) - 1), label="return")
        ax.plot(100*(np.exp(log_returns['p_market_return'].cumsum()) - 1), label="p_market_return")
        ax.plot(100 * (np.exp(log_returns['p_minVar_return'].cumsum()) - 1), label="p_minVar_return")
        ax.plot(100 * (np.exp(log_returns['p_maxShp_return'].cumsum()) - 1), label="p_maxShp_return")
        ax.legend(loc="best")
        ax.tick_params(rotation=90)
        ax.set_ylabel('Total Relative Returns (%)')

        plt.suptitle('train dates:' + start_date + " - " + end_train_date)
        plt.tight_layout()
        plt.show()

        fig = plt.figure()
        plt.bar(sharpe.index, sharpe.to_numpy())
        plt.suptitle('train dates:' + start_date + " - " + end_train_date)
        plt.tight_layout()
        plt.show()

    return sharpe


def main() -> pd.DataFrame:

    FROM_DATE_ = '2015-01-01'
    TO_DATE_ = '2022-07-30'

    all_data = get_data(FROM_DATE_, 'NA', TO_DATE_)

    #2021
    _START_DATE_ = ['2015-12-01', '2016-01-01', '2016-02-01', '2016-03-01', '2016-04-01', '2016-05-01', '2016-06-01', '2016-07-01', '2016-08-01', '2016-09-01', '2016-10-01', '2016-11-01']
    _END_TRAIN_DATE_ = ['2020-12-31', '2021-01-31', '2021-02-28', '2021-03-31', '2021-04-30', '2021-05-31', '2021-06-30', '2021-07-31', '2021-08-31', '2021-09-30', '2021-10-31', '2021-11-30']
    _END_TEST_DATE_ = ['2021-01-31', '2021-02-28', '2021-03-31', '2021-04-30', '2021-05-31', '2021-06-30', '2021-07-31', '2021-08-31', '2021-09-30', '2021-10-31', '2021-11-30', '2021-12-31']

    params = [('train_minVar', None, 0), ('train_minVar', 300, 0), ('train_minVar', 300, 0.1),
              ('train_maxShp', None, 0), ('train_maxShp', 300, 0) , ('train_maxShp', 300, 0.1),
              ('train_equal', None)]

    results = dict.fromkeys([*params, 'minVar', 'market', 'maxShp'])

    for key, _ in results.items():
        results[key] = [0] * len(_START_DATE_)

    for p in params:
        print("params = ", p)

        for i in tqdm(range(len(_START_DATE_))):

            shrp = test_portfolio(_START_DATE_[i], _END_TRAIN_DATE_[i], _END_TEST_DATE_[i], data=all_data, params=p)

            results[p][i] = shrp['return']
            results['minVar'][i] = shrp['p_minVar_return']
            results['market'][i] = shrp['p_market_return']
            results['maxShp'][i] = shrp['p_maxShp_return']

    df_results = pd.DataFrame(results)
    return df_results

if __name__ == '__main__':

    results = main()
    #results.to_pickle('../results.pkl')
    #plt.plot(results, label=results.columns)
    #plt.legend(loc="best")
    #plt.show()
