from pathlib import Path
import datetime
from itertools import product, chain
from collections import ChainMap
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, Akima1DInterpolator, PchipInterpolator
# import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

from joblib import Parallel, delayed

# %matplotlib inline
DATA_PATH = Path("input").expanduser()
train_path = Path('train.csv')
test_path = Path('Track 1/test.csv')

target_keys = ["max", "min"]
target_keys += [f"p{p:0.2}" for p in np.arange(0.04, 1.0, 0.04)]
target_keys += [f"pp{p:0.2}" for p in np.arange(0.04, 1.0, 0.04)]

START_DATE = pd.to_datetime("2014-01-01 00:00:00")
END_DATE = pd.to_datetime("2014-03-01 00:00:00")
max_elapsed_sec = int((END_DATE - START_DATE).total_seconds() * 1.2)


def predict(param_dict):
    # pred_idx_list = np.arange(param_dict["n_wave_min"],
    #                           int(param_dict["n_wave_max"] * 5))
    fit_res1 = param_dict[f"max_1"]
    m_root = (np.poly1d(fit_res1) - max_elapsed_sec).roots
    # print(len(m_root))
    if len(m_root) == 2:
        m1, m2 = m_root[0], m_root[1]
        if m1 < param_dict["n_wave_min"] < m2:
            m = m2
        elif m2 < param_dict["n_wave_min"] < m1:
            m = m1
        elif param_dict["n_wave_min"] < m1 < m2:
            m = m1
        elif param_dict["n_wave_min"] < m2 < m1:
            m = m2
    elif len(m_root) == 1:
        m = m_root[0]
    m = int(m * 1.7)

    res_list = []
    for key in target_keys:
        fit_res1 = param_dict[f"{key}_1"]
        fit_res2 = param_dict[f"{key}_2"]
        pred_idx_list = np.arange(param_dict["n_wave_min"], m)
        # print(param_dict["n_wave_min"], m, np.poly1d(fit_res1)(m))

        pred_esec_list = np.poly1d(fit_res1)(pred_idx_list)
        pred_target_list = np.poly1d(fit_res2)(pred_esec_list)
        res_p_df = pd.DataFrame({
            "elapsed_sec": pred_esec_list,
            "target": pred_target_list
        })
        res_list.append(res_p_df)

    pred_df = pd.concat(res_list).sort_values("elapsed_sec")
    return pred_df


def make_submission(pred_df, test_esec_list, type = 2):
    if type == 0:
        fitted_curve = interp1d(pred_df["elapsed_sec"],
                                pred_df["target"],
                                kind='linear')

    if type == 1:
        fitted_curve = interp1d(pred_df["elapsed_sec"],
                                pred_df["target"],
                                kind='quadratic')

    if type == 2:
        fitted_curve = Akima1DInterpolator(pred_df["elapsed_sec"],
                                           pred_df["target"])

    if type == 3:
        fitted_curve = PchipInterpolator(pred_df["elapsed_sec"], pred_df["target"])

    return fitted_curve(test_esec_list)


def test_predict():
    sat_id = 443
    target_col = 'x'
    tmp_df = predict(fit_param_dict[(sat_id, target_col)])


def main():
    TARGET_COLS = ["x", "y", "z", "Vx", "Vy", "Vz"]
    SIM_COLS = [col + "_sim" for col in TARGET_COLS]

    with open('fit_param_dictAA.dump', 'rb') as reader:
        fit_param_dict = pickle.load(reader)

    fit_param_dict[(443, 'x')]

    # load test data
    test = pd.read_csv(DATA_PATH / test_path, index_col=0)
    test.epoch = pd.to_datetime(test.epoch)
    test["elapsed_sec"] = test["epoch"] - START_DATE
    test["elapsed_sec"] = test["elapsed_sec"].apply(
        lambda x: x.total_seconds())
    A_SAT_ID_LIST = test["sat_id"].unique()

    # prepare submission df
    submission = test[SIM_COLS].copy()
    submission.columns = TARGET_COLS
    submission.tail()

    # extrapolate
    # akima
    for i_sat, (sat_id, sat_df) in tqdm(enumerate(test.groupby('sat_id'))):
        # print(sat_id)
        # sat_id = 1
        test_idx = sat_df.index

        for target_col in TARGET_COLS:
            # target_col = 'x'
            pred_df = predict(fit_param_dict[(sat_id, target_col)])
            # print(pred_df["elapsed_sec"].min(), pred_df["elapsed_sec"].max())
            # print(sat_df["elapsed_sec"].min(), sat_df["elapsed_sec"].max())
            if target_col in ['x','y','z']:
                val = make_submission(pred_df, sat_df["elapsed_sec"].values, type=2)
            elif target_col in ['Vx','Vy','Vz']:
                val = make_submission(pred_df, sat_df["elapsed_sec"].values, type=2)

            submission.loc[test_idx, target_col] = val

    submission.tail()
    # submission_akima.to_csv(RESULT_PATH/'submission_akima.csv')
    submission_akima = submission.copy()

    # prepare submission df
    submission = test[SIM_COLS].copy()
    submission.columns = TARGET_COLS
    submission.tail()

    # extrapolate
    # pchip
    for i_sat, (sat_id, sat_df) in tqdm(enumerate(test.groupby('sat_id'))):
        # print(sat_id)
        # sat_id = 1
        test_idx = sat_df.index

        for target_col in TARGET_COLS:
            # target_col = 'x'
            pred_df = predict(fit_param_dict[(sat_id, target_col)])
            # print(pred_df["elapsed_sec"].min(), pred_df["elapsed_sec"].max())
            # print(sat_df["elapsed_sec"].min(), sat_df["elapsed_sec"].max())
            if target_col in ['x','y','z']:
                val = make_submission(pred_df, sat_df["elapsed_sec"].values, type=3)
            elif target_col in ['Vx','Vy','Vz']:
                val = make_submission(pred_df, sat_df["elapsed_sec"].values, type=3)

            submission.loc[test_idx, target_col] = val

    submission.tail()
    # submission.to_csv(RESULT_PATH/'submission_pchip.csv')
    submission_pchip = submission.copy()

    # submission_akima = pd.read_csv('submission_akima.csv',index_col=0)
    # submission_pchip = pd.read_csv('submission_pchip.csv',index_col=0)

    r = [1,1,0,0]
    submission = (r[0]*submission_akima+r[1]*submission_pchip)/sum(r)
    submission.to_csv('submission_trackA.csv', index=True)

if __name__ == '__main__':
    main()
