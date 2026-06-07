import numpy as np
import pandas as pd
import calendar
from datetime import datetime
from dateutil.relativedelta import relativedelta


def get_last_date(year, month):
     last_day = calendar.monthrange(year, month)[1]
     return f"{last_day:02d}/{month:02d}/{year}"


min_strike = 1
max_strike = 19
stepsize_k = 0.05
n_strikes = int((max_strike - min_strike) / stepsize_k + 1)
strikes = np.linspace(min_strike, max_strike, n_strikes, endpoint=True)
strikes_formatted = [f'{k:.0f}' if k.is_integer() else f'{k:.1f}' for k in strikes]

max_duration_trade_months = 26 #months
range_end_year_trades = range(20, 28)  #forward looking
range_months_trades = range(1, 13)

today = datetime.today().strftime('%Y-%m-%d')
df_lseg_requests = pd.DataFrame(columns=['Trades', 'DataType', 'Start', 'End'])
i = 0


for year in range_end_year_trades:
    for month in range_months_trades:
        mm = str(month)
        digits = len(mm)
        if digits < 2:
            mm = f'0{mm}'

        mmyy = f'{mm}{year}'
        t0 = pd.to_datetime(f'01/{mm}/20{year}', dayfirst=True)
        start_trade_date = t0 - relativedelta(months=max_duration_trade_months)
        last_trade_date = t0 + relativedelta(months=1)
        start_date = f'01/{start_trade_date.month:02d}/{start_trade_date.year}'
        end_date = get_last_date(last_trade_date.year, last_trade_date.month)

        puts = [f'LNE{mmyy}{k}P' for k in strikes_formatted]
        calls = [f'LNE{mmyy}{k}C' for k in strikes_formatted]
        trades = puts + calls

        df_lseg_requests.loc[i, 'Trades'] = ','.join(trades)
        df_lseg_requests.loc[i, 'DataType'] = 'VL'
        df_lseg_requests.loc[i, 'Start'] = start_date
        if pd.to_datetime(end_date, dayfirst=True) <= pd.to_datetime(today, dayfirst=True):
            df_lseg_requests.loc[i, 'End'] = end_date

        i +=1

df_lseg_requests.to_excel('lseg_request_table.xlsx')

