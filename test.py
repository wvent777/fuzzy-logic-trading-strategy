import skfuzzy as fuzz
import numpy as np
from skfuzzy import control as ctrl
import warnings
import sys
import os.path
warnings.filterwarnings('ignore')
fast = np.array([30, 30, 10, 1])
slow = np.array([10, 20, 50, 20])
print('Fast Moving Average\n', fast)
print('Slow Moving Average\n', slow)

def splitSeven(fastMA:np.array, slowMA:np.array)->np.array:
    # Positive when Bullish # Negative When Bearish
    difference = fastMA - slowMA
    print('Difference of Moving Averages\n', difference)
    max_point = max(difference)
    min_point = min(difference)
    pos_part = 0
    if max_point > 0:
        pos_part = np.ceil(max_point / 3)
    neg_part = 0
    if min_point < 0:
        neg_part = np.floor(min_point / 3)
    return [neg_part * 3,
            neg_part * 2,
            neg_part,
            0,
            pos_part,
            pos_part * 2,
            pos_part * 3]

test =  splitSeven(fast, slow)
print('Adjusted to 7 extents\n', test)


def buildFuzzySystem(arr, bShowPlot=False, bDebug=False):
    '''Given an array of value, build and return a fuzzy system'''
    # New Antecedent/Consequent objects hold universe variables and membership functions

    feModel = ctrl.Antecedent(np.arange(arr[0], arr[-1], 0.1), 'feModel')
    recModel = ctrl.Consequent(np.arange(-1, 1, 0.1), 'recModel')

    recModel["Neg"] = fuzz.pimf(recModel.universe, -1, -1, -1, 0)
    recModel["Med"] = fuzz.pimf(recModel.universe, -1, 0, 0, 1)
    recModel["Pos"] = fuzz.pimf(recModel.universe, 0, 1, 1, 1)

    feModel["EL"] = fuzz.pimf(feModel.universe, arr[0], arr[0], arr[0], arr[1])
    feModel["VL"] = fuzz.pimf(feModel.universe, arr[0], arr[1], arr[1], arr[2])
    feModel["L"] = fuzz.pimf(feModel.universe, arr[1], arr[2], arr[2], arr[3])
    feModel["M"] = fuzz.pimf(feModel.universe, arr[2], arr[3], arr[3], arr[4])
    feModel["H"] = fuzz.pimf(feModel.universe, arr[3], arr[4], arr[4], arr[5])
    feModel["VH"] = fuzz.pimf(feModel.universe, arr[4], arr[5], arr[5], arr[6])
    feModel["EH"] = fuzz.pimf(feModel.universe, arr[5], arr[6], arr[6], arr[6])

    if bShowPlot:
        feModel.view() # Antecedent
        recModel.view() # Consequent


    if bDebug:
        print(type(feModel))
        print(type(recModel))

    # build the rules
    rule1 = ctrl.Rule(feModel["EL"] | feModel["L"] | feModel["VL"], recModel["Neg"])
    rule2 = ctrl.Rule(feModel["M"], recModel["Med"])
    rule3 = ctrl.Rule(feModel["EH"] | feModel["H"] | feModel["VH"], recModel["Pos"])

    ctrl_sys = ctrl.ControlSystem([rule1, rule2, rule3])
    ctrl_instance = ctrl.ControlSystemSimulation(ctrl_sys)

    return ctrl_instance, feModel, recModel


def getFuzzyOutput(ctrl_instance, feModel, recModel, input_val, bShowPlot=False, bDebug=False):
    ''' Given the fuzzy system and the input value, compute and return the mf and the fuzzy extent'''

    ext = ""
    rec = 0
    ants = []
    try:
        ctrl_instance.input['feModel'] = input_val
        ctrl_instance.compute()

        if bShowPlot:
            feModel.view(sim=ctrl_instance)
            recModel.view(sim=ctrl_instance)

        # ants = ctrl_instance.get_antecedents()
        ants = ctrl_instance.input
        if bDebug:
            print(type(feModel))
            print(type(recModel))

            # print (ctrl_instance.print_state())

        ext = ""
        if ants:
            ext = ants
            rec = ctrl_instance.output['recModel']
    except:
        print("Input val:{}. fe:{}, rec:{}, ctrl:{}".format(input_val, feModel, recModel, type(ctrl_instance)))
        print("Type ants:{} and content:{}".format(type(ants), ants))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("Exception type:{}, File name:{} and line no:{}".format(exc_type, fname, exc_tb.tb_lineno))
        # TODO: Handle this error properly
        raise

    return (rec, ext)


def getFuzzyExtent(farr, val):
    ''' Given the values for the extent zones, return the matching extent name'''
    exts = ['VL', 'EL', 'L', 'M', 'H', 'VH', 'EH']
    zones = list(zip(farr, exts))

    # print (zones[0][0])

    for i, f in enumerate(zones):
        if val > zones[i - 1][0] and val < zones[i][0]:
            # print ("My ext is:{}".format(f[1]))
            return f[1]


diffval = -25
fuzsys, fe, rec = buildFuzzySystem(test, bShowPlot=True)
mf, ext = getFuzzyOutput(fuzsys, fe, rec, diffval, bShowPlot=True)
print('Exampel Difference Value:{}'.format(diffval))
print("Membership:{}, Extent Value:{}".format(mf, ext))