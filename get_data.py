import yfinance as yf

tickers = ['FXE', 'EWJ', 'GLD', 'QQQ', 'SPY', 'SHV', 'DBA', 'USO', 'XBI', 'ILF', 'EZA', 'EPP', 'FEZ']
prices = yf.download(tickers, start='2018-01-01', end='2023-01-01')['Adj Close']

returns = prices / prices.shift(1) - 1
returns = returns.dropna()

returns.to_csv('5yr_ret_data.csv')

