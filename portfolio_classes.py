import pandas as pd
import numpy as np
from scipy.optimize import LinearConstraint, Bounds, minimize
from sklearn import linear_model


def tgt_beta_obj(w, exp_rets, w_p, lam):
    """
    Computes the OBJ for a max return portfolio with rebalancing debit. Target beta constraint imposed
    in scipy.optimize
    :param w: Current Portfolio Weights
    :param exp_rets: Vector of Expected Returns for Each Asset
    :param w_p: Prior Portfolio Weights
    :param lam: Penalty factor for rebalancing portfolio
    :return: Objective function value, to be minimized
    """
    d_t = 1/2/lam * exp_rets + w_p
    obj = -d_t.transpose().dot(w) + 1/2 * w.transpose().dot(w)
    return obj


def mark_obj(w, w_p, lam, cov_mtrx):
    """
    Computes the OBJ for a min variance portfolio with rebalancing costs. Target return constraint imposed
    in scipy.optimize
    :param w: Current Portfolio Weights
    :param w_p: Prior Portfolio Weights
    :param lam: Penalty factor for rebalancing portfolio
    :param cov_mtrx: Covariance Matrix for Assets in Universe
    :return: Objective function value, to be minimized
    """
    diff = w - w_p
    obj = w.transpose().dot(cov_mtrx).dot(w) + lam * diff.transpose().dot(diff)
    return obj


def get_betas(returns):
    output = returns.cov() / returns['SPY'].var()
    return output.loc['SPY']


def get_mark_weights(cov_mtrx, exp_rets, w_p, lam, return_tgt):
    # Bound weights in individual assets on [-2, 2]
    bounds = Bounds(-2 * np.ones(len(exp_rets)), 2 * np.ones(len(exp_rets)))

    # Define constraints for optimization:
    # 1. Fully Invested (Sum of all holdings is 1)
    # 2. Expected Portfolio Return equals target (15%)
    daily_ret_tgt = (1+return_tgt) ** (1/252) - 1
    cons = np.vstack((np.ones(len(exp_rets)), exp_rets))
    lb = [1, daily_ret_tgt]
    ub = [1, daily_ret_tgt]
    linear_constraint = LinearConstraint(cons, lb, ub)

    w_0 = pd.DataFrame(np.ones(len(exp_rets)) / len(exp_rets), index=exp_rets.index)[0]

    opt = minimize(mark_obj, w_0, method='trust-constr', args=(w_p, lam, cov_mtrx),
                   constraints=[linear_constraint], bounds=bounds)

    output = pd.DataFrame(opt.x, index=exp_rets.index)[0]
    return output


def get_betaport_weights(exp_rets, w_p, lam, betas, beta_tgt):
    # Bound weights in individual assets on [-2, 2]
    bounds = Bounds(-2 * np.ones(len(exp_rets)), 2 * np.ones(len(exp_rets)))

    # Define constraints for optimization
    # 1. Fully Invested (Sum of all holdings is 1)
    # 2. Expected Portfolio Beta equals target
    cons = np.vstack((np.ones(len(exp_rets)), betas))
    lb = [1, beta_tgt]
    ub = [1, beta_tgt]
    linear_constraint = LinearConstraint(cons, lb, ub)

    w_0 = pd.DataFrame(np.ones(len(exp_rets)) / len(exp_rets), index=exp_rets.index)[0]

    opt = minimize(tgt_beta_obj, w_0, method='trust-constr', args=(exp_rets, w_p, lam),
                   constraints=[linear_constraint], bounds=bounds)

    output = pd.DataFrame(opt.x, index=exp_rets.index)[0]
    return output


def ff_exp_return(hist_returns, hist_ff):
    factor_df = pd.DataFrame(data=None, columns=['Bi3', 'bis', 'biv', 'alpha'])
    x = hist_ff[['Mkt-RF', 'SMB', 'HML']]
    for stock in hist_returns.columns:
        y = hist_returns[stock] - hist_ff['RF']
        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        factor_df.loc[stock] = np.append(regr.coef_, regr.intercept_)

    last_obs = hist_ff.iloc[-1].drop('RF')
    last_obs.index = ['Bi3', 'bis', 'biv']
    last_obs.loc['alpha'] = 1

    output = last_obs.dot(factor_df.transpose()) + hist_ff.iloc[-1]['RF']
    return output


class Markowitz_Strategy:
    def __init__(self, hist_returns, ff_factors, ret_period, risk_period, lam, tgt_ret):
        self.hist_returns = hist_returns
        self.ff_factors = ff_factors
        self.ret_period = ret_period
        self.risk_period = risk_period
        self.lam = lam
        self.tgt_ret = tgt_ret

        self.holdings = pd.DataFrame(columns=hist_returns.columns)

        max_count = max(self.ret_period, self.risk_period)
        w_p = np.ones(len(hist_returns.columns)) / len(hist_returns.columns)

        start = ff_factors.index[max_count].date()
        print(f'Simulating Markowitz Strategy from [{start} : {ff_factors.index[-1].date()}]')
        print(f'Return Period: {ret_period}     Risk Period: {risk_period}')
        print(f'Rebalancing Penalty: {lam}     Target Return: {tgt_ret}')
        print('...')

        for i in range(ff_factors.shape[0] - max_count + 1):
            if risk_period > ret_period:
                risk_ret_sub = hist_returns.iloc[(0 + i):(max_count + i)]
                ff_ret_sub = hist_returns.iloc[(max_count - ret_period + i):(max_count + i)]
                fama_sub = ff_factors.iloc[(max_count - ret_period + i):(max_count + i)]
            else:
                ff_ret_sub = hist_returns.iloc[(0 + i):(max_count + i)]
                fama_sub = ff_factors.iloc[(0 + i):(max_count + i)]
                risk_ret_sub = hist_returns.iloc[(max_count - risk_period + i):(max_count + i)]

            # Get Covariance Matrix
            # Get Expected Returns
            # Get Markowitz Portfolio Weights
            # Update w_p to be latest solution
            # After running, shift solutions back one timestep

            exp_rets = ff_exp_return(ff_ret_sub, fama_sub)

            weights = get_mark_weights(cov_mtrx=risk_ret_sub.cov(), exp_rets=exp_rets,
                                       w_p=w_p, lam=self.lam, return_tgt=self.tgt_ret)

            w_p = weights
            self.holdings.loc[ff_ret_sub.index[-1]] = weights

        self.holdings = self.holdings.shift(1).dropna()
        self.daily_port_ret = (hist_returns.loc[self.holdings.index] * self.holdings).sum(axis=1)
        self.avg_ret = self.daily_port_ret.mean()
        self.pnl = (1+self.daily_port_ret).cumprod() - 1

        self.dd10 = round((self.pnl - self.pnl.shift(10)).min() * -1, 4)
        self.vol = round(self.daily_port_ret.std() * 252 ** (1/2), 4)
        self.annual_ret = round((self.avg_ret + 1) ** 252 - 1, 4)

        # Get average monthly RF rate (Fama Factor) for Sharpe Ratio
        avg_rf_rate = self.ff_factors['RF'].loc[self.holdings.index].mean()
        self.sharpe = round(((self.avg_ret - avg_rf_rate + 1) ** 252 - 1) / self.vol, 4)

        mkt_return = self.hist_returns['SPY'].loc[self.holdings.index].mean()
        mkt_return = round((1 + mkt_return) ** 252 - 1, 4)

        print(f'Max 10 Day Drawdown:  {self.dd10}')
        print(f'Annualized Return:  {self.annual_ret}')
        print(f'Annualized Volatility:  {self.vol}')
        print(f'Sharpe Ratio:  {self.sharpe}')
        print(f'Market Return: {mkt_return}')

        print('')
        print('Simulation Complete')
        print('')


class Beta_Tgt_Strategy:
    def __init__(self, hist_returns, ff_factors, ret_period, risk_period, lam, beta_tgt):
        self.hist_returns = hist_returns
        self.ff_factors = ff_factors
        self.ret_period = ret_period
        self.risk_period = risk_period
        self.lam = lam
        self.beta_tgt = beta_tgt

        self.holdings = pd.DataFrame(columns=hist_returns.columns)

        max_count = max(self.ret_period, self.risk_period)
        w_p = np.ones(len(hist_returns.columns)) / len(hist_returns.columns)

        start = ff_factors.index[max_count].date()
        print(f'Simulating Target Beta Strategy from [{start} : {ff_factors.index[-1].date()}]')
        print(f'Return Period: {ret_period}     Risk Period: {risk_period}')
        print(f'Rebalancing Penalty: {lam}     Target Beta: {beta_tgt}')
        print('...')

        for i in range(ff_factors.shape[0] - max_count + 1):
            if risk_period > ret_period:
                risk_ret_sub = hist_returns.iloc[(0 + i):(max_count + i)]
                ff_ret_sub = hist_returns.iloc[(max_count - ret_period + i):(max_count + i)]
                fama_sub = ff_factors.iloc[(max_count - ret_period + i):(max_count + i)]
            else:
                ff_ret_sub = hist_returns.iloc[(0 + i):(max_count + i)]
                fama_sub = ff_factors.iloc[(0 + i):(max_count + i)]
                risk_ret_sub = hist_returns.iloc[(max_count - risk_period + i):(max_count + i)]

            # Get Expected Returns
            # Get Asset Betas
            # Get Target Beta Portfolio Weights
            # Update w_p to be latest solution
            # After running, shift solutions back one timestep

            exp_rets = ff_exp_return(ff_ret_sub, fama_sub)
            betas = get_betas(risk_ret_sub)

            weights = get_betaport_weights(exp_rets=exp_rets, w_p=w_p, lam=self.lam,
                                           betas=betas, beta_tgt=self.beta_tgt)

            w_p = weights
            self.holdings.loc[ff_ret_sub.index[-1]] = weights

        self.holdings = self.holdings.shift(1).dropna()
        self.daily_port_ret = (hist_returns.loc[self.holdings.index] * self.holdings).sum(axis=1)
        self.avg_ret = self.daily_port_ret.mean()
        self.pnl = (1+self.daily_port_ret).cumprod() - 1

        self.dd10 = round((self.pnl - self.pnl.shift(10)).min() * -1, 4)
        self.vol = round(self.daily_port_ret.std() * 252 ** (1/2), 4)
        self.annual_ret = round((self.avg_ret + 1) ** 252 - 1, 4)

        # Get average monthly RF rate (Fama Factor) for Sharpe Ratio
        avg_rf_rate = self.ff_factors['RF'].loc[self.holdings.index].mean()
        self.sharpe = round(((self.avg_ret - avg_rf_rate + 1) ** 252 - 1) / self.vol, 4)

        mkt_return = self.hist_returns['SPY'].loc[self.holdings.index].mean()
        mkt_return = round((1+mkt_return) ** 252 - 1, 4)

        print(f'Max 10 Day Drawdown:  {self.dd10}')
        print(f'Annualized Return:  {self.annual_ret}')
        print(f'Annualized Volatility:  {self.vol}')
        print(f'Sharpe Ratio:  {self.sharpe}')
        print(f'Market Return: {mkt_return}')

        print('')
        print('Simulation Complete')
        print('')


if __name__ == '__main__':
    return_data = pd.read_csv('5yr_ret_data.csv', parse_dates=True, index_col='Date')['2019']
    fama_data = pd.read_csv('3fac_fama.csv', parse_dates=True, index_col='Date')['2019']

    test = Markowitz_Strategy(return_data, fama_data, ret_period=100, risk_period=200, lam=100, tgt_ret=0.20)

    test = Beta_Tgt_Strategy(return_data, fama_data, ret_period=100, risk_period=200, lam=100, beta_tgt=1.5)

    print('DONE')

