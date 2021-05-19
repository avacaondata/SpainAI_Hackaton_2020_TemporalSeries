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


def get_submission_markowitz(weights, assets):
    weights = _fix_weights(weights)
    subm_plantilla = pd.read_csv("./submission/submission.csv")
    date = subm_plantilla["eod_ts"]
    cols = {
        f"allo_{asset}": [weights[f"{asset}_close"]] * len(date) for asset in assets
    }
    return pd.DataFrame({"eod_ts": date, **cols})
