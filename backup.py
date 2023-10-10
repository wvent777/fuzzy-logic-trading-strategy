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
    return np.array(tradeRes)

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
               fitnesstocalc: np.array)-> np.array:
    for i in fitnesstocalc:
        fitness[i, 1] = stratFitness(priceData, strats[str(i)])
    return fitness

def stratFitness(priceData: list, strat: dict, testing:bool=False)-> float:
    fitness = []
    for df in priceData:
        dfStrat = getMAcol(deepcopy(df), strat)
        dfStrat = dfStrat[dfStrat['Date'] > LOWER_DATE]

        # Run Strat
        tradeRes = runStrat(dfStrat['Open'].values.astype(np.float64),
                            dfStrat['fast'].values.astype(np.float64),
                            dfStrat['slow'].values.astype(np.float64))

        if STRAT_EVAL == 'mean':
            fitnessVal = np.mean(tradeRes)
        elif STRAT_EVAL == 'median':
            fitnessVal = np.median(tradeRes)
        elif STRAT_EVAL == 'compounded':
            fitnessVal = getCompounded(tradeRes)
        else:
            raise ValueError('Invalid STRAT_EVAL')

        if tradeRes.shape[0] > MIN_TRADES or testing:
            fitness.append(fitnessVal)
        else:
            fitness.append(0)
    if FITNESS_TYPE=='min':
        return np.min(fitness)
    elif FITNESS_TYPE=='mean':
        return np.mean(fitness)
    elif FITNESS_TYPE=='median':
        return np.median(fitness)
    else:
        raise ValueError('Invalid FITNESS_TYPE')

@nb.jit(nopython=True)
def getCompounded(tradeRes: np.array):
    ''' Get compounded return '''
    invest = 1
    for perc in tradeRes:
        invest = (1+perc)*invest
    return invest



def main() -> dict:
    ''' main ga script
    returns dict: optimized strategy parameters'''

    priceData, strats, fitness, fitnesstocalc = initGA()

    # Num of strategies to change per generation
    numToChange = int(NUM_STRATS * (1 - KEEP_BEST))

    fitnessSave = []
    for gen in range(0, NUM_GENERATIONS):
        fitness = getFitness(priceData, strats, fitness, fitnesstocalc)

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

        print(f'Generation {gen}')
        for count, strat in enumerate(np.flipud(goodStrats[-5:])):
            print(
                str(count) + '. Strategy: ' + str(strat) +
                ', ' + FITNESS_TYPE + ': ' +
                str(fitness[strat, 1])
            )
        print('----------------------------------------------')
        fitnessSave.append(deepcopy(fitness))

    return strats[str(goodStrats[-1])], fitnessSave, goodStrats

if __name__ == '__main__':

    TRAINING_TICKERS = ['NVDA', 'SPY', 'GOOG', 'TSLA', 'AAPL']
    NUM_STRATS = 50
    KEEP_BEST = 0.3
    NUM_GENERATIONS = 150
    MA_TYPES = ['simple', 'exponential']
    MA_FIELDS = ['Open', 'Low', 'High', 'Close', "Adj Close"]
    LOWER_MA_PERIOD = 3  # minimum of ma
    UPPER_MA_PERIOD = 300  # maximum of ma
    MAX_PERTURB = 10  # Max number to perturb the strategy parameters with (adds/subtract and swaps ma type)
    LOWER_DATE = '2017-01-01'
    STARTING_STRAT = {
        'fast_ma_type': 'simple',
        'slow_ma_type': 'simple',
        'fast_ma_field': 'Close',
        'slow_ma_field': 'Close',
        'fast_ma_period': 5,
        'slow_ma_period': 20}
    STRAT_EVAL = 'compounded'
    FITNESS_TYPE = 'mean'
    MIN_TRADES = 20
    TESTING_TICKERS = ['AMZN', 'QQQ']
    t0 = time.time()
    strat, fitness, bestStrats = main()
    print(f'Optimization time: {time.time() - t0}')
    # Run the strategy and baseline on the testing tickers
    priceData = [pd.read_csv(f'Data/{ticker}.csv') for ticker in TESTING_TICKERS]

    baseline = stratFitness(priceData, STARTING_STRAT, testing=True)
    optimized = stratFitness(priceData, strat, testing=True)

    print('\n')
    print(f'Testing Values Baseline: {baseline}')
    print(f'Testing Values Optimized: {optimized}')
    print('\nOptimal Strategy Parameters:', strat)

    best_strats = {strat:[] for strat in bestStrats[0:2]}
    for gen in range(len(fitness)):
        for strat in bestStrats[0:2]:
            best_strats[strat].append(fitness[gen][strat][1])

    bestpd = pd.DataFrame(best_strats)
    print(bestpd)
    fig = px.line(bestpd, y=bestpd.columns, x=bestpd.index, log_x=True)
    fig.show()