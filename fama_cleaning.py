import pandas as pd

raw = pd.read_csv('raw_fama.csv')

clean_df = raw
clean_df['Unnamed: 0'] = pd.to_datetime(raw['Unnamed: 0'], format='%Y%m%d')
clean_df.columns = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RF']
clean_df = clean_df.set_index('Date')
clean_df = clean_df['2018-01-01':'2023-01-01']
clean_df[['Mkt-RF', 'SMB', 'HML', 'RF']] = clean_df[['Mkt-RF', 'SMB', 'HML', 'RF']]/100

returns = pd.read_csv('5yr_ret_data.csv')
returns['Date'] = pd.to_datetime(returns['Date'], format='%Y-%m-%d')
returns = returns.set_index('Date')

no_rets = list(set(clean_df.index) - set(returns.index))

clean_df.drop(no_rets, inplace=True)

clean_df.to_csv('3fac_fama.csv')

