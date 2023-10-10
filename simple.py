import numpy as np
import pandas as pd
from typing import Tuple
import plotly.io as pio
import plotly.graph_objects as go
STRATEGY = {
    'fast_ma_type': 'simple',
    'slow_ma_type': 'simple',
    'fast_ma_field': 'Close',
    'slow_ma_field': 'Close',
    'fast_ma_period': 5,
    'slow_ma_period': 20,
}


def getMAcol(df: pd.DataFrame, strat:dict) -> pd.DataFrame:
    ''' Add moving average column to dataframe as specified by strategy '''

    for ma_type in ['slow', 'fast']:

        if strat[ma_type + '_ma_type'] == 'simple':
            df[ma_type] = df[strat[ma_type + '_ma_field']].\
                rolling(strat[ma_type + '_ma_period']).mean()

        elif strat[ma_type + '_ma_type'] == 'exponential':
            df[ma_type] = df[strat[ma_type + '_ma_field']].\
                ewm(span=strat[ma_type + '_ma_period'], adjust=False).mean()
        else:
            raise ValueError('MA type not supported')
    return df

def runStrat(openPrices: np.array, fastMA: np.array, slowMA: np.array) -> np.array:

    ''' Run moving average strategy, buy day after the fast ma crosses the slow ma and
    vice versa
    returns -> tradeRes: np.array percentage of gained/lost on each trade
    '''

    # if currently holding asset
    holding = False

    tradeRes = []
    boughtOn=[]
    soldOn=[]

    # The logical criteria for if a ma crossover happens, both on the buy and
    # sell side
    maBuy = lambda day: (
        fastMA[day-2] < slowMA[day-2] and
        fastMA[day-1] > slowMA[day-1]
    )
    maSell = lambda day: (
        fastMA[day-2] > slowMA[day-2] and
        fastMA[day-1] < slowMA[day-1]
    )

    for day in range(2, len(openPrices)):
        if not holding and maBuy(day):
            boughtAt = openPrices[day]
            print(day, 'buying at', boughtAt)
            boughtOn.append(day)
            holding = True

        elif holding and maSell(day):
            tradeRes.append(openPrices[day]/boughtAt - 1)
            print(day, 'selling at', openPrices[day], 'result', tradeRes[-1])
            soldOn.append(day)
            holding=False

    # only care about closed trades
    if holding:
        boughtOn.pop()
    return np.array(boughtOn), np.array(soldOn), np.array(tradeRes)

def getEquityCurve(df: pd.DataFrame, df_backtest:pd.DataFrame, init_invest:float) -> Tuple[list, list]:
    invest_val = init_invest
    equity_curve=[]
    dates= []

    for trade in range(len(df_backtest)):
        df_trade = df[
            (df['Date'] >= df_backtest.loc[trade, 'bought_on']) &
            (df['Date'] <= df_backtest.loc[trade, 'sold_on'])]
        equity = invest_val*df_trade['Close'].values/df_trade['Close'].values[0]

        equity_curve += list(equity)
        dates += list(df_trade['Date'])
        invest_val = equity[-1]
    return dates, equity_curve

def plotEquityCurve(TICKER, init_invest: float, plot_buy_hold:bool, lower_date:str, title:str):
    df = pd.read_csv('Data/' + TICKER + '.csv')
    df_backtest = pd.read_csv(f'{TICKER}_{NAME}_backtest.csv')

    df = df[df['Date'] >= lower_date]
    df = df.reset_index(drop=True)

    df_backtest = df_backtest[df_backtest['bought_on'] >= lower_date]
    df_backtest = df_backtest.reset_index(drop=True)

    fig = go.Figure()
    if plot_buy_hold:
        buy_hold_equity = init_invest*df['Close'].values/df['Close'].values[0]

        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=buy_hold_equity,
            name='Buy and Hold',
            line={'color': 'rgba(241, 128, 48, 1)'}))
    else:
        buy_hold_equity = []
    dates, equity_curve = getEquityCurve(df, df_backtest, init_invest)
    fig.add_trace(go.Scatter(
        x=dates,
        y=equity_curve,
        name='Strategy',
        line={'color': 'rgba(48, 128, 241, 1)'}))

    min_y = min(min(buy_hold_equity), min(equity_curve))
    max_y = max(max(buy_hold_equity), max(equity_curve))

    for trade in range(len(df_backtest)):
        if df_backtest.loc[trade, 'profit'] > 0:
            color = 'rgba(0, 236, 109, 0.2)'
        else:
            color = 'rgba(255, 0, 0, 0.2)'

        fig.add_shape(
            type='rect',
            x0 = df_backtest.loc[trade, 'bought_on'],
            x1 = df_backtest.loc[trade, 'sold_on'],
            y0 = min_y,
            y1 = max_y,
            line = {'color': 'rgba(0, 0, 0, 0)'},
            fillcolor = color,
            layer = 'below'
        )

    fig.update_layout(
        title=title,
        xaxis={'title': 'Date'},
        yaxis={'title': 'Equity($)', 'range': [min_y, max_y]},
        legend={'orientation': 'h', 'x': 0, 'y': 1.075},
        width=800,
        height=700,
    )
    fig.show()
    return fig


if __name__ =='__main__':
    TICKER = 'AAPL'
    NAME = 'baseline_strategy'
    LOWER_DATE = '2017-01-01'

    df = pd.read_csv('Data/' + TICKER + '.csv')
    df = getMAcol(df, STRATEGY)

    # apply lower date filter
    df = df[df['Date'] >= LOWER_DATE]

    # Run Strat
    boughtON, soldON, tradeRes = runStrat(
        df['Open'].values.astype(np.float64),
        df['fast'].values.astype(np.float64),
        df['slow'].values.astype(np.float64))
    dates = df['Date'].values

    df_backtest = pd.DataFrame({
        'bought_on': dates[boughtON],
        'sold_on': dates[soldON],
        'profit': tradeRes,
    })
    df_backtest.to_csv(f'{TICKER}_{NAME}_backtest.csv', index=False)

    print(df_backtest)
    plotEquityCurve(TICKER, 1000, plot_buy_hold=True, lower_date=LOWER_DATE, title='APPLE MA Strategy')