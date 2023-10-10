import random
from copy import deepcopy
import numba as nb
import numpy as np
import pandas as pd
from typing import Tuple
import time
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from numpy import ndarray
from datetime import datetime as dt

state = random.getstate()
random.setstate(state)

def getMAcol(df: pd.DataFrame, strat:dict) -> pd.DataFrame:
    ''' Add moving average column to dataframe as specified by strategy '''

    for ma_type in ['slow', 'fast']:

        if strat[ma_type + '_ma_type'] == 'simple':
            df[ma_type] = df[strat[ma_type + '_ma_field']].\
                rolling(strat[ma_type + '_ma_period']).mean()
        elif strat[ma_type + '_ma_type'] == 'exponential':
            df[ma_type] = df[strat[ma_type + '_ma_field']].\
                ewm(span=strat[ma_type + '_ma_period'], adjust=False).mean()
        elif strat[ma_type + '_ma_type'] == 'weighted':
            temp_weights = list(np.random.dirichlet(np.ones(strat[ma_type + '_ma_period']), size=1)[0])
            weights = sorted(temp_weights, reverse=True)
            df[ma_type] = df[strat[ma_type + '_ma_field']].\
                rolling(strat[ma_type + '_ma_period']).apply(lambda x: np.dot(x, weights)/sum(weights))
        else:
            raise ValueError('MA type not supported')
    return df

@nb.jit(nopython=True)
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
            # print(day, 'buying at', boughtAt)
            boughtOn.append(day)
            holding = True

        elif holding and maSell(day):
            tradeRes.append(openPrices[day]/boughtAt - 1)
            # print(day, 'selling at', openPrices[day], 'result', tradeRes[-1])
            soldOn.append(day)
            holding=False

    # only care about closed trades
    if holding:
        boughtOn.pop()

    return np.array(tradeRes), np.array(boughtOn), np.array(soldOn)

def getRandomStrat() -> dict:
    ''' Generate a random strategy '''
    return checkStrat({
        'fast_ma_type': random.choice(MA_TYPES),
        'slow_ma_type': random.choice(MA_TYPES),
        'fast_ma_field': random.choice(MA_FIELDS),
        'slow_ma_field': random.choice(MA_FIELDS),
        'fast_ma_period': random.randint(LOWER_MA_PERIOD, UPPER_MA_PERIOD),
        'slow_ma_period': random.randint(LOWER_MA_PERIOD, UPPER_MA_PERIOD)})


def checkStrat(strat: dict) -> dict:
    "Checks if strat has valid parameters, adjusts if not"
    for ma_type in ['slow', 'fast']:
        if strat[ma_type + '_ma_period'] < LOWER_MA_PERIOD:
            strat[ma_type + '_ma_period'] = LOWER_MA_PERIOD
        elif strat[ma_type + '_ma_period'] > UPPER_MA_PERIOD:
            strat[ma_type + '_ma_period'] = UPPER_MA_PERIOD
    if strat['slow_ma_period'] <= strat['fast_ma_period']:
        strat['slow_ma_period'] = strat['fast_ma_period'] + 1
    return strat


def perturbStrat(strat: dict) -> dict:
    for ma_type in ['slow', 'fast']:
        strat[ma_type + '_ma_type'] = random.choice(MA_TYPES)
        strat[ma_type + '_ma_field'] = random.choice(MA_FIELDS)
        strat[ma_type + '_ma_period'] += random.randint(-MAX_PERTURB, MAX_PERTURB)
    return checkStrat(strat)

def breedWinningStrat(goodStrats: np.array, strats:dict)->dict:
    ''' Breed a winning strategy from the top 30% of the population '''
    newStrat = {}
    for key in strats['0'].keys():
        randomStratIdx = str(random.choice(goodStrats))
        newStrat[key] = strats[randomStratIdx][key]
    return checkStrat(newStrat)

def initGA()-> Tuple[list, dict, np.array, np.array]:
    ''' Initialize the genetic algorithm
     priceData: list
     strats: dict
     fitness: np.array
     fitnesstocalc: np.array
     '''

    priceData = [pd.read_csv(f'Data/{ticker}.csv')
    for ticker in TRAINING_TICKERS]

    strats = {f'{n}' : perturbStrat(deepcopy(STARTING_STRAT)) for n in range(NUM_STRATS)}

    fitness = np.zeros((NUM_STRATS, 2))
    fitness[:, 0] = np.arange(NUM_STRATS)

    fitnesstocalc = np.arange(NUM_STRATS)
    return priceData, strats, fitness, fitnesstocalc

def getFitness(priceData: list,
               strats: dict,
               fitness: np.array,
               fitnesstocalc: np.array,
               strat_eval,
               fitness_type)-> np.array:
    for i in fitnesstocalc:
        fitness[i, 1] = stratFitness(priceData, strats[str(i)], strat_eval, fitness_type)
    return fitness

def stratFitness(priceData: list, strat: dict, strat_eval:str, fitness_type:str, testing:bool=False)-> ndarray:
    fitness = []
    for df in priceData:
        dfStrat = getMAcol(deepcopy(df), strat)
        dfStrat = dfStrat[dfStrat['Date'] > LOWER_DATE]

        # Run Strat
        tradeRes, _, _ = runStrat(dfStrat['Open'].values.astype(np.float64),
                            dfStrat['fast'].values.astype(np.float64),
                            dfStrat['slow'].values.astype(np.float64))

        if strat_eval == 'mean':
            fitnessVal = np.mean(tradeRes)
        elif strat_eval == 'median':
            fitnessVal = np.median(tradeRes)
        elif strat_eval == 'compounded':
            fitnessVal = getCompounded(tradeRes)
        else:
            raise ValueError('Invalid STRAT_EVAL')

        if tradeRes.shape[0] > MIN_TRADES or testing:
            fitness.append(fitnessVal)
        else:
            fitness.append(0)
    if fitness_type == 'min':
        return np.min(fitness)
    elif fitness_type == 'mean':
        return np.mean(fitness)
    elif fitness_type == 'median':
        return np.median(fitness)
    elif fitness_type == 'max':
        return np.max(fitness)
    else:
        raise ValueError('Invalid FITNESS_TYPE')

@nb.jit(nopython=True)
def getCompounded(tradeRes: np.array):
    ''' Get compounded return '''
    invest = 1
    for perc in tradeRes:
        invest = (1+perc)*invest
    return invest



def main(strat_eval, fitness_type, verbose=False) -> dict:
    ''' main ga script
    returns dict: optimized strategy parameters'''

    priceData, strats, fitness, fitnesstocalc = initGA()

    # Num of strategies to change per generation
    numToChange = int(NUM_STRATS * (1 - KEEP_BEST))

    fitnessSave = []
    for gen in range(0, NUM_GENERATIONS):
        fitness = getFitness(priceData, strats, fitness, fitnesstocalc, strat_eval, fitness_type)

        ranks = fitness[fitness[:, 1].argsort()]
        goodStrats = ranks[numToChange:, 0].astype(np.int32)
        badStrats = ranks[:numToChange, 0].astype(np.int32)

        splits = np.array_split(badStrats, 3)

        # replace bad strategies with new
        for strat in splits[0]:
            strats[str(strat)] = getRandomStrat()

        # add random strats
        for strat in splits[1]:
            randStrat = str(random.choice(goodStrats))
            strats[str(strat)] = perturbStrat(deepcopy(strats[randStrat]))

        for strat in splits[2]:
            strats[str(strat)] = breedWinningStrat(goodStrats, deepcopy(strats))

        fitnesstocalc = badStrats

        if verbose:
            print(f'Generation {gen}')
            for count, strat in enumerate(np.flipud(goodStrats[-5:])):
                print(
                    str(count) + '. Strategy: ' + str(strat) +
                    ', ' + fitness_type + ': ' +
                    str(fitness[strat, 1])
                )
            print('----------------------------------------------')
            fitnessSave.append(deepcopy(fitness))
        fitnessSave.append(deepcopy(fitness))

    return strats[str(goodStrats[-1])], fitnessSave, goodStrats

def eval(ticker:str, strategy: dict, name: str)->pd.DataFrame:
    df = pd.read_csv(f'Data/{ticker}.csv')
    dfStrat = getMAcol(deepcopy(df), strategy)
    dfStrat = dfStrat[dfStrat['Date'] > LOWER_DATE]
    tradeRes, buy, sell = runStrat(dfStrat['Open'].values.astype(np.float64),
                                   dfStrat['fast'].values.astype(np.float64),
                                   dfStrat['slow'].values.astype(np.float64))
    dates = dfStrat['Date'].values
    df_backtest = pd.DataFrame({
        'Bought_On': dates[buy],
        'Sold_On': dates[sell],
        'Profit': tradeRes
    })
    df_backtest.to_csv(f'Data/{ticker}_{name}_strategy_backtest.csv', index=False)
    return df_backtest

def getEquityCurve(df: pd.DataFrame, df_backtest:pd.DataFrame, init_invest:float) -> Tuple[list, list]:
    invest_val = init_invest
    equity_curve=[]
    dates= []

    for trade in range(len(df_backtest)):
        df_trade = df[
            (df['Date'] >= df_backtest.loc[trade, 'Bought_On']) &
            (df['Date'] <= df_backtest.loc[trade, 'Sold_On'])]
        equity = invest_val*df_trade['Adj Close'].values/df_trade['Adj Close'].values[0]

        equity_curve += list(equity)
        dates += list(df_trade['Date'])
        invest_val = equity[-1]
    return dates, equity_curve

def plotEquityCurve(ticker, init_invest: float, plot_buy_hold:bool, lower_date:str, title:str):
    df = pd.read_csv('Data/' + ticker + '.csv')
    df_backtest_baseline = pd.read_csv(f'Data/{ticker}_baseline_strategy_backtest.csv')
    df_max = pd.read_csv(f'Data/{ticker}_max_strategy_backtest.csv')
    df_min = pd.read_csv(f'Data/{ticker}_min_strategy_backtest.csv')
    df_mean = pd.read_csv(f'Data/{ticker}_mean_strategy_backtest.csv')
    df_median = pd.read_csv(f'Data/{ticker}_median_strategy_backtest.csv')

    df = df[df['Date'] >= lower_date]
    df = df.reset_index(drop=True)

    df_backtest_baseline = df_backtest_baseline[df_backtest_baseline['Bought_On'] >= lower_date]
    df_backtest_baseline = df_backtest_baseline.reset_index(drop=True)

    df_max = df_max[df_max['Bought_On'] >= lower_date]
    df_max = df_max.reset_index(drop=True)

    df_min = df_min[df_min['Bought_On'] >= lower_date]
    df_min = df_min.reset_index(drop=True)

    df_mean = df_mean[df_mean['Bought_On'] >= lower_date]
    df_mean = df_mean.reset_index(drop=True)

    df_median = df_median[df_median['Bought_On'] >= lower_date]
    df_median = df_median.reset_index(drop=True)
    fig = go.Figure()
    if plot_buy_hold:
        buy_hold_equity = init_invest*df['Adj Close'].values/df['Adj Close'].values[0]

        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=buy_hold_equity,
            name='Buy and Hold',
            line={'color': 'rgba(241, 128, 48, 1)'}))
    else:
        buy_hold_equity = []
    dates_base, equity_curve_baseline = getEquityCurve(df, df_backtest_baseline, init_invest)
    dates_max, equity_curve_max = getEquityCurve(df, df_max, init_invest)
    dates_min, equity_curve_min = getEquityCurve(df, df_min, init_invest)
    dates_mean, equity_curve_mean = getEquityCurve(df, df_mean, init_invest)
    dates_med, equity_curve_median = getEquityCurve(df, df_median, init_invest)

    fig.add_trace(go.Scatter(
        x=dates_base,
        y=equity_curve_baseline,
        name='Baseline Strategy',
        line={'color': 'cyan'}))

    fig.add_trace(go.Scatter(
        x=dates_max,
        y=equity_curve_max,
        name='Optimized Strategy (Max)',
        line={'color': 'green'}))

    fig.add_trace(go.Scatter(
        x=dates_min,
        y=equity_curve_min,
        name='Optimized Strategy (Min)',
        line={'color': 'red'}))

    fig.add_trace(go.Scatter(
        x=dates_mean,
        y=equity_curve_mean,
        name='Optimized Strategy (Mean)',
        line={'color': 'yellow'}))

    fig.add_trace(go.Scatter(
        x=dates_med,
        y=equity_curve_median,
        name='Optimized Strategy (Median)',
        line={'color': 'purple'}))


    min_y = min(min(buy_hold_equity), min(equity_curve_baseline), min(equity_curve_max), min(equity_curve_min),
                min(equity_curve_mean), min(equity_curve_median))
    max_y = max(max(buy_hold_equity), max(equity_curve_baseline), max(equity_curve_max), max(equity_curve_min),
                max(equity_curve_mean), max(equity_curve_median))
    # for trade in range(len(df_backtest)):
    #     if df_backtest.loc[trade, 'profit'] > 0:
    #         color = 'rgba(0, 236, 109, 0.2)'
    #     else:
    #         color = 'rgba(255, 0, 0, 0.2)'
    #
    #     fig.add_shape(
    #         type='rect',
    #         x0 = df_backtest.loc[trade, 'bought_on'],
    #         x1 = df_backtest.loc[trade, 'sold_on'],
    #         y0 = min_y,
    #         y1 = max_y,
    #         line = {'color': 'rgba(0, 0, 0, 0)'},
    #         fillcolor = color,
    #         layer = 'below'
    #     )
    fig.update_layout(
        title=title,
        xaxis={'title': 'Date'},
        yaxis={'title': 'Equity($)', 'range': [min_y, max_y]},
        legend={'orientation': 'h', 'x': 0, 'y': 1.075},
        width=1000,
        height=800,
    )
    fig.show()
    return fig

if __name__ == '__main__':

    TRAINING_TICKERS = ['NVDA', 'AMZN', 'GOOG', 'TSLA', 'TSM']
    NUM_STRATS = 40
    KEEP_BEST = 0.3
    NUM_GENERATIONS = 20
    MA_TYPES = ['simple', 'exponential', 'weighted']
    MA_FIELDS = ['Open', 'Low', 'High', 'Close', "Adj Close"]
    LOWER_MA_PERIOD = 3  # minimum of ma
    UPPER_MA_PERIOD = 200  # maximum of ma
    MAX_PERTURB = 10  # Max number to perturb the strategy parameters with (adds/subtract and swaps ma type)
    LOWER_DATE = '2017-01-01'
    STARTING_STRAT = {
        'fast_ma_type': 'simple',
        'slow_ma_type': 'simple',
        'fast_ma_field': 'Adj Close',
        'slow_ma_field': 'Adj Close',
        'fast_ma_period': 5,
        'slow_ma_period': 20}
    MIN_TRADES = 30
    TESTING_TICKERS = ['META', 'GOOGL', 'AAPL']
    t0 = time.time()

    # Run the baseline strategy
    for ticker in TESTING_TICKERS+TRAINING_TICKERS:
        eval(ticker, STARTING_STRAT, 'baseline')

    mean_strat, mean_fitness, mean_bestStrats = main(strat_eval='compounded', fitness_type='mean')
    max_strat, max_fitness, max_bestStrats = main(strat_eval='compounded', fitness_type='max')
    median_strat, median_fitness, median_bestStrats = main(strat_eval='compounded', fitness_type='median')
    min_strat, min_fitness, min_bestStrats = main(strat_eval='compounded', fitness_type='min')

    strat_names = ['mean', 'max', 'median', 'min']
    for strat, name in zip([mean_strat, max_strat, median_strat, min_strat], strat_names):
        for ticker in TESTING_TICKERS:
            eval(ticker, strat, name)

    print(f'Optimization time: {time.time() - t0}')
    # Run the strategy and baseline on the testing tickers
    priceData = [pd.read_csv(f'Data/{ticker}.csv') for ticker in TESTING_TICKERS]
    baseline = stratFitness(priceData, STARTING_STRAT, strat_eval='compounded', fitness_type='mean', testing=True)
    mean_optimized = stratFitness(priceData, mean_strat, strat_eval='compounded', fitness_type='mean', testing=True)
    max_optimized = stratFitness(priceData, max_strat, strat_eval='compounded', fitness_type='max', testing=True)
    median_optimized = stratFitness(priceData, median_strat, strat_eval='compounded', fitness_type='median', testing=True)
    min_optimized = stratFitness(priceData, min_strat, strat_eval='compounded', fitness_type='min', testing=True)

    print('\n')
    print(f'Testing Values Baseline: {baseline}')
    print(f'Testing Values Mean Optimized: {mean_optimized}')
    print(f'Testing Values Max Optimized: {max_optimized}')
    print(f'Testing Values Median Optimized: {median_optimized}')
    print(f'Testing Values Min Optimized: {min_optimized}')

    print('\nOptimal Mean Strategy Parameters:', mean_strat)
    print('Optimal Max Strategy Parameters:', max_strat)
    print('Optimal Median Strategy Parameters:', median_strat)
    print('Optimal Min Strategy Parameters:', min_strat)

    combined = {'mean': [], 'max': [], 'median': [], 'min': []}
    for gen in range(len(min_fitness)):
        combined['mean'].append(mean_fitness[gen][mean_bestStrats[-1]][1])
        combined['max'].append(max_fitness[gen][max_bestStrats[-1]][1])
        combined['median'].append(median_fitness[gen][median_bestStrats[-1]][1])
        combined['min'].append(min_fitness[gen][min_bestStrats[-1]][1])

    combined_df = pd.DataFrame(combined)

    fig = px.line(combined_df, x=combined_df.index, y=combined_df.columns,
                  labels={'index': 'Generation', 'value': 'Fitness'}, symbol='variable',
                  title='Fitness of Best Strategies Per Generation')
    fig['data'][0]['line']['color'] = 'yellow'
    fig['data'][1]['line']['color'] = 'green'
    fig['data'][2]['line']['color'] = 'purple'
    fig['data'][3]['line']['color'] = 'red'
    fig.show()

    fig.write_image('Images/fitness.jpg')
    metaeq = plotEquityCurve('META', 10000, plot_buy_hold=True, lower_date=LOWER_DATE, title='META Equity Curves')
    googeq = plotEquityCurve('GOOGL', 10000, plot_buy_hold=True, lower_date=LOWER_DATE, title='GOOGL Equity Curves')
    aapleq = plotEquityCurve('AAPL', 10000, plot_buy_hold=True, lower_date=LOWER_DATE, title='AAPL Equity Curves')
    metaeq.write_image('Images/metaeq.jpg')
    aapleq.write_image('Images/aapleq.jpg')
    googeq.write_image('Images/googeq.jpg')
