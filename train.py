# %load_ext autoreload
# %autoreload 2

import datetime
from pathlib import Path
from itertools import product, chain
from collections import ChainMap
import pandas as pd
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

from joblib import Parallel, delayed


DATA_PATH = Path('input').expanduser()
train_path = Path('train.csv')
test_path = Path('Track 1/test.csv')


def smape(satellite_predicted_values, satellite_true_values):
    # the division, addition and subtraction are pointwise
    return np.mean(
        np.abs((satellite_predicted_values - satellite_true_values) /
               (np.abs(satellite_predicted_values) +
                np.abs(satellite_true_values))))


def add_time_feature(train, test, TEST_SAT_ID_LIST):
    train_list = {}
    test_list = {}
    for sat_id in tqdm(range(600)):
        train_sliced = train[train["sat_id"] == sat_id].copy()
        train_sliced = train_sliced.sort_values("epoch")
        train_len = len(train_sliced)
        train_sliced["is_test"] = 0
        train_sliced["is_valid"] = 0
        p_valid = 0.5
        train_sliced.loc[train_sliced.index[int(len(train_sliced) *
                                                p_valid):], "is_valid"] = 1
        if sat_id in TEST_SAT_ID_LIST:
            test_sliced = test[test["sat_id"] == sat_id].copy()
            test_sliced["is_valid"] = 0
            test_sliced["is_test"] = 1
            train_sliced = pd.concat([train_sliced, test_sliced],
                                     axis=0,
                                     sort=False)

        train_sliced = train_sliced.sort_values("epoch")
        train_sliced["elapsed_sec"] = (
            train_sliced["epoch"] -
            train_sliced["epoch"].iloc[0]).apply(lambda x: x.total_seconds())
        epoch = train_sliced["epoch"]

        tmp_list = [0]
        num_small_delt = 0
        for i in range(1, len(train_sliced)):
            if (epoch.iloc[i] - epoch.iloc[i - 1]).total_seconds() < 1:
                num_small_delt += 1

            tmp_list.append(i - num_small_delt)
        train_sliced["time_idx"] = tmp_list

        train_list[sat_id] = train_sliced.iloc[:train_len]
        if sat_id in TEST_SAT_ID_LIST:
            test_list[sat_id] = train_sliced.iloc[train_len:]

    train = pd.concat(train_list.values())
    test = pd.concat(test_list.values())
    return train, test


def naibun(v1, v2, m, n):
    return (n * v1 + m * v2) / (m + n)


def linear_reg(idx_list, esec_list, target_list, test_esec_max, d1=1, d2=1):
    fit_res1 = np.polyfit(idx_list, esec_list, d1)
    fit_res2 = np.polyfit(esec_list, target_list, d2)
    pred_idx_list = [len(esec_list) - 2, len(esec_list) - 1]
    pred_idx = len(esec_list)
    while True:
        pred_idx_list.append(pred_idx)
        if np.poly1d(fit_res1)(pred_idx) > test_esec_max:
            break
        pred_idx += 1
    pred_esec_list = np.poly1d(fit_res1)(pred_idx_list)
    pred_target_list = np.poly1d(fit_res2)(pred_esec_list)
    return fit_res1, fit_res2, pred_idx_list, pred_esec_list, pred_target_list


def linear_reg_mini(idx_list, esec_list, target_list, d1=1, d2=1):
    fit_res1 = np.polyfit(idx_list, esec_list, d1)
    fit_res2 = np.polyfit(esec_list, target_list, d2)
    return fit_res1, fit_res2


def fit(train_sliced, valid_sliced, target_col, d1=1, d2=1):
    fit_res_dict = {}

    # get maximum & minimum points
    maximum_idx = []
    minimum_idx = []
    for i in range(1, len(train_sliced) - 1):
        if train_sliced[target_col].iloc[i] > train_sliced[target_col].iloc[i-1] and \
            train_sliced[target_col].iloc[i] > train_sliced[target_col].iloc[i+1]:
            maximum_idx.append(i)
        if train_sliced[target_col].iloc[i] < train_sliced[target_col].iloc[i-1] and \
            train_sliced[target_col].iloc[i] < train_sliced[target_col].iloc[i+1]:
            minimum_idx.append(i)

    extremum_idx = maximum_idx + minimum_idx
    extremum_idx.sort()

    # linear regression for maximum points
    idx_list = np.arange(len(maximum_idx))
    esec_list = train_sliced["elapsed_sec"].iloc[maximum_idx]
    target_list = train_sliced[target_col].iloc[maximum_idx]
    fit_res1, fit_res2 = linear_reg_mini(idx_list,
                                         esec_list,
                                         target_list,
                                         d1=d1,
                                         d2=d2)
    fit_res_dict["max_1"] = fit_res1
    fit_res_dict["max_2"] = fit_res2
    n_wave_min = len(esec_list) - 2
    n_wave_max = int(idx_list[-1] * 5)

    # linear regression for minimum points
    idx_list = np.arange(len(minimum_idx))
    esec_list = train_sliced["elapsed_sec"].iloc[minimum_idx]
    target_list = train_sliced[target_col].iloc[minimum_idx]
    fit_res1, fit_res2 = linear_reg_mini(idx_list,
                                         esec_list,
                                         target_list,
                                         d1=d1,
                                         d2=d2)
    fit_res_dict["min_1"] = fit_res1
    fit_res_dict["min_2"] = fit_res2
    n_wave_min = min(n_wave_min, len(esec_list) - 2)
    n_wave_max = max(n_wave_max, int(idx_list[-1] * 2.3))

    # linear regression for points between maximum and minimum points
    for p in np.arange(0.04, 1.0, 0.04):
        epoch_list = []
        target_list = []
        for i in range(1, len(extremum_idx)):
            tmp_df = train_sliced.iloc[extremum_idx[i - 1]:extremum_idx[i] + 1]
            tmp_df = tmp_df.sort_values(target_col)
            target = train_sliced[target_col].iloc[extremum_idx[
                i - 1]] * p + train_sliced[target_col].iloc[
                    extremum_idx[i]] * (1 - p)
            target_list.append(target)
            tmp1 = tmp_df[tmp_df[target_col] <= target].iloc[-1]
            tmp2 = tmp_df[tmp_df[target_col] > target].iloc[0]
            e1 = tmp1["elapsed_sec"]
            e2 = tmp2["elapsed_sec"]
            t1 = tmp1[target_col]
            t2 = tmp2[target_col]
            e = naibun(e1, e2, target - t1, t2 - target)
            epoch_list.append(e)

        idx_list0 = np.arange(len(epoch_list[0::2]))
        esec_list0 = epoch_list[0::2]
        target_list0 = target_list[0::2]
        fit_res1, fit_res2 = linear_reg_mini(idx_list0,
                                             esec_list0,
                                             target_list0,
                                             d1=d1,
                                             d2=d2)
        fit_res_dict[f"p{p:0.2}_1"] = fit_res1
        fit_res_dict[f"p{p:0.2}_2"] = fit_res2
        n_wave_min = min(n_wave_min, len(esec_list) - 2)
        n_wave_max = max(n_wave_max, int(idx_list[-1] * 2.3))

        idx_list1 = np.arange(len(epoch_list[1::2]))
        esec_list1 = epoch_list[1::2]
        target_list1 = target_list[1::2]
        fit_res1, fit_res2 = linear_reg_mini(idx_list1,
                                             esec_list1,
                                             target_list1,
                                             d1=d1,
                                             d2=d2)
        fit_res_dict[f"pp{p:0.2}_1"] = fit_res1
        fit_res_dict[f"pp{p:0.2}_2"] = fit_res2
        n_wave_min = min(n_wave_min, len(esec_list) - 2)
        n_wave_max = max(n_wave_max, int(idx_list[-1] * 2.3))

    fit_res_dict["n_wave_min"] = n_wave_min
    fit_res_dict["n_wave_max"] = n_wave_max
    return fit_res_dict


def get_param_dict():
    param_dict = {}
    for sat_id in range(600):
        for target_col in TARGET_COLS:
            param_dict[(sat_id, target_col)] = {
                "d1": 1,
                "d2": 1,
                "valid_r": 0.5
            }

    param_df = pd.read_csv('lr_hyperparam.csv')
    for i in range(len(param_df)):
        sat_id = param_df['sat_id'].iloc[i]
        target_col = param_df['target_col'].iloc[i]
        d1 = param_df['d1'].iloc[i]
        d2 = param_df['d2'].iloc[i]
        valid_r = param_df['valid_r'].iloc[i]
        if sat_id == 478:
            continue

        param_dict[(sat_id, target_col)] = {
            "d1": d1,
            "d2": d2,
            "valid_r": valid_r
        }

    return param_dict


def main():
    # load data
    train = pd.read_csv(DATA_PATH / train_path, index_col=0)
    test = pd.read_csv(DATA_PATH / test_path, index_col=0)
    train.epoch = pd.to_datetime(train.epoch)
    test.epoch = pd.to_datetime(test.epoch)

    TEST_SAT_ID_LIST = test["sat_id"].unique()
    B_SAT_ID_LIST = sorted(set(range(600)) - set(TEST_SAT_ID_LIST))

    # add time_idx and elapsed sec
    train.head()
    train, test = add_time_feature(train, test, TEST_SAT_ID_LIST)

    # load parameters
    param_dict = get_param_dict()

    # parallel
    def process(sat_id):
        train_sliced = train[(train["sat_id"] == sat_id)
                             & (train["is_valid"] == 0)].copy()
        train_sliced = train_sliced.drop_duplicates("time_idx")
        train_sliced = train_sliced.sort_values("epoch")

        valid_sliced = train[(train["sat_id"] == sat_id)
                             & (train["is_valid"] == 1)].copy()
        valid_sliced = valid_sliced.drop_duplicates("time_idx")
        valid_sliced = valid_sliced.sort_values("epoch")

        res_dict = {}

        for target_col in TARGET_COLS:
            d1 = param_dict[(sat_id, target_col)]["d1"]
            d2 = param_dict[(sat_id, target_col)]["d2"]
            valid_r = param_dict[(sat_id, target_col)]["valid_r"]

            if len(valid_sliced) < 72:
                tmp_valid = pd.concat([train_sliced, valid_sliced])
            else:
                tmp_valid = pd.concat([train_sliced, valid_sliced])
                if valid_r < 1:
                    tmp_valid = tmp_valid.iloc[-int(valid_r * len(tmp_valid)):]
                else:
                    tmp_valid = tmp_valid.iloc[-int(valid_r):]

            fit_param_dict = fit(
                tmp_valid,
                None,  #test_sliced,
                target_col,
                d1=d1,
                d2=d2)
            res_dict[(sat_id, target_col)] = fit_param_dict

            res_dict[(sat_id, target_col)]["d1"] = d1
            res_dict[(sat_id, target_col)]["d2"] = d2
            res_dict[(sat_id, target_col)]["valid_r"] = valid_r

        return res_dict

    # for trackA
    res_listA = Parallel(n_jobs=-1, verbose=10)([
        delayed(process)(sat_id)
        # for sat_id in B_SAT_ID_LIST
        for sat_id in TEST_SAT_ID_LIST
    ])
    fit_param_dictA = ChainMap(*res_listA)

    with open('fit_param_dictAA.dump', 'wb') as writer:
        pickle.dump(fit_param_dictA, writer)

    # for trackB
    res_listB = Parallel(n_jobs=-1, verbose=10)(
        [delayed(process)(sat_id) for sat_id in B_SAT_ID_LIST])
    fit_param_dictB = ChainMap(*res_listB)

    with open('fit_param_dictB2.dump', 'wb') as writer:
        pickle.dump(fit_param_dictB, writer)


if __name__ == '__main__':
    TARGET_COLS = ["x", "y", "z", "Vx", "Vy", "Vz"]
    SIM_COLS = [col + "_sim" for col in TARGET_COLS]
    main()
