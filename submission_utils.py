import numpy as np
import pandas as pd


def _fix_weights(weights):
    suma_weights = sum(weights.values())
    if suma_weights < 1.0:
        # print(sum(weights.values()))
        max_weight_asset = list(sorted(weights, key=weights.get, reverse=True))[0]
        falta = 1 - suma_weights
        weights[max_weight_asset] += falta
        # print(sum(weights.values()))
        assert sum(weights.values()) == 1.0
        return weights
    elif suma_weights > 1.0:
        max_weight_asset = list(sorted(weights, key=weights.get, reverse=True))[0]
        sobra = suma_weights - 1
        weights[max_weight_asset] -= sobra
        assert sum(weights.values()) == 1.0
        assert all([v > 0 for v in weights.values()])
    return weights


def general_weights_fixer(lista):
    while sum(lista) != 1:
        lista = np.clip(lista, 0, 1)
        lista = [round(act, 4) for act in lista]
        suma = sum(lista)
        if suma > 1:
            sobra = suma - 1
            idx = np.argmax(lista)
            lista[idx] -= sobra
        elif suma < 1:
            falta = 1 - suma
            idx = np.argmax(lista)
            lista[idx] += falta
    return lista


def get_submission_markowitz(weights, assets):
    subm_plantilla = pd.read_csv("./submission/submission.csv")
    date = subm_plantilla["eod_ts"]
    cols = {
        f"allo_{asset.replace('_close', '')}": [weights[f"{asset}"]] * len(date)
        for asset in assets
    }
    df = pd.DataFrame({"eod_ts": date, **cols})
    df.set_index("eod_ts", inplace=True)
    return df


def test_submission(submission):
    if "eod_ts" in submission.columns:
        submission.set_index("eod_ts", inplace=True)
    assert all(submission.sum(axis=1) == 1)
    assert sum(n < 0 for n in submission.values.flatten()) == 0


def _weights_to_dict(weights, asset_names):
    return {asset: weight for asset, weight in zip(asset_names, weights)}


def _convert_daily_submission(submission, plantilla):
    df_index = pd.DataFrame({"date": plantilla.index}).set_index("date")
    final_submission = pd.merge(
        submission.resample("H").aggregate("mean").ffill(),
        df_index,
        right_index=True,
        left_index=True,
        how="right",
    ).ffill()
    final_submission.index = final_submission.index.rename("eod_ts")
    final_submission = final_submission.div(final_submission.sum(axis=1), axis=0)
    final_submission.columns = [f"allo_{col}" for col in final_submission.columns]
    return final_submission
