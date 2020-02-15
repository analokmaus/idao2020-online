import datetime
from itertools import product, chain
from collections import ChainMap
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, Akima1DInterpolator, PchipInterpolator
from tqdm import tqdm
import pickle

# import matplotlib.pyplot as plt
# %matplotlib inline

p_list = [
    0.04, 0.08, 0.12, 0.2, 0.24, 0.32, 0.40, 0.48, 0.56, 0.64, 0.72, 0.80,
    0.88, 0.92, 0.96
]
target_keys = ["max", "min"]
target_keys += [f"p{p:0.2}" for p in p_list]
target_keys += [f"pp{p:0.2}" for p in p_list]

START_DATE = pd.to_datetime("2014-01-01 00:00:00")
END_DATE = pd.to_datetime("2014-03-01 00:00:00")
max_elapsed_sec = int((END_DATE - START_DATE).total_seconds() * 1.2)


def predict(param_dict):
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

        # linear regression :)
        pred_esec_list = np.poly1d(fit_res1)(pred_idx_list)
        pred_target_list = np.poly1d(fit_res2)(pred_esec_list)
        res_p_df = pd.DataFrame({
            "elapsed_sec": pred_esec_list,
            "target": pred_target_list
        })
        res_list.append(res_p_df)

    pred_df = pd.concat(res_list).sort_values("elapsed_sec")
    return pred_df


def make_submission(pred_df, test_esec_list):
    # linear-interpolation
    fitted_curve = interp1d(pred_df["elapsed_sec"],
                            pred_df["target"],
                            kind='linear')

    # fitted_curve = interp1d(pred_df["elapsed_sec"],
    #                         pred_df["target"],
    #                         kind='quadratic')

    # fitted_curve = Akima1DInterpolator(pred_df["elapsed_sec"],
    #                                    pred_df["target"])

    # fitted_curve = PchipInterpolator(pred_df["elapsed_sec"], pred_df["target"])

    return fitted_curve(test_esec_list)


# def test_predict():
#     sat_id = 0
#     target_col = 'x'
#     tmp_df = predict(fit_param_dict[(sat_id, target_col)])


def main():
    TARGET_COLS = ["x", "y", "z", "Vx", "Vy", "Vz"]
    SIM_COLS = [col + "_sim" for col in TARGET_COLS]

    # load parameters of linear regressions
    with open('fit_param_dictB2.dump', 'rb') as reader:
        fit_param_dict = pickle.load(reader)

    # load test data
    test = pd.read_csv('test.csv', index_col=0)
    test.epoch = pd.to_datetime(test.epoch)
    test["elapsed_sec"] = test["epoch"] - START_DATE
    test["elapsed_sec"] = test["elapsed_sec"].apply(
        lambda x: x.total_seconds())
    B_SAT_ID_LIST = test["sat_id"].unique()

    # prepare submission df
    submission = test[SIM_COLS].copy()
    submission.columns = TARGET_COLS

    # calc prediction
    for sat_id in B_SAT_ID_LIST:
        # print(sat_id)
        sat_df = test[test["sat_id"] == sat_id].sort_values("epoch")
        test_idx = sat_df.index

        for target_col in TARGET_COLS:
            # extrapolate
            pred_df = predict(fit_param_dict[(sat_id, target_col)])
            # interpolate
            val = make_submission(pred_df, sat_df["elapsed_sec"].values)

            submission.loc[test_idx, target_col] = val

    submission.to_csv('submission.csv', index=True)


main()
