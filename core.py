from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def rmsle(actual, pred):
    pred[pred < 0] = 0
    return mean_squared_log_error(actual, pred) ** 0.5


def make_sub(preds, path):
    preds[preds < 0] = 0
    sub = pd.read_csv('../data/sampleSubmission.csv')
    sub['count'] = preds
    sub.to_csv(path, index=False)


def eval_model(model, X, y):
    print(cross_val_score(model, X, y, scoring=make_scorer(rmsle)).mean())


def eval_data(model, path=None, path_test=None, path_out=None,
              ds=None, ds_test=None, export_test_set=False,
              evaluate=True, target_transform_fn=None, rounds=1):
    if ds is None:
        ds = pd.read_csv(path)

    X, y = ds.drop('count', axis=1), ds['count']

    if evaluate:
        print(np.array(
            [cross_val_score(model, X, y, scoring=make_scorer(rmsle), n_jobs=-1).mean() for i in range(rounds)]
        ).mean())

    if export_test_set and \
            path_out is not None and \
            (path_test is not None or ds_test is not None):
        if ds_test is None:
            ds_test = pd.read_csv(path_test)
        model.fit(X, y)
        preds = model.predict(ds_test)

        if target_transform_fn is not None:
            preds = target_transform_fn(preds)
        make_sub(preds, path_out)


def corr(ds):
    return abs(ds.corr()['count']).sort_values()


def handle_outliers(df_raw, columns, drop=False):
    df = df_raw.copy()

    for column in columns:
        if column not in df: continue

        upper_lim = df[column].quantile(.95)
        lower_lim = df[column].quantile(.05)

        if not drop:
            df.loc[(df[column] > upper_lim), column] = upper_lim
            df.loc[(df[column] < lower_lim), column] = lower_lim
        else:
            df = df.loc[(df[column] < upper_lim) & (df[column] > lower_lim)]

    return df


def transform(df_raw, columns, fn=np.log1p):
    df = df_raw.copy()
    for column in columns:
        if column in df:
            df[column] = df[column].transform(fn)

    return df


def drop(df_raw, columns):
    df = df_raw.copy()
    for column in columns:
        if column in df:
            df.drop(column, axis=1, inplace=True)

    return df


def groupby_mean(df_raw, pairs):
    df = df_raw.copy()
    for group_col, agr_col in pairs:
        df = pd.merge(df, df.groupby(group_col)[agr_col].mean(),
                      left_on=group_col, right_on=group_col, suffixes=('', f'_{group_col}_mean'))
    return df


def scale(df_raw, columns, minMax=False):
    df = df_raw.copy()

    if minMax:
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    for col in columns:
        if col in df_raw:
            df[col] = scaler.fit_transform(np.array(df[col]).reshape(-1, 1)).reshape(-1)
    return df


def create_rf_model(trial):
    return RandomForestRegressor(
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 15),
        min_samples_split=trial.suggest_uniform("min_samples_split", 0.05, 1.0),
        n_estimators=trial.suggest_int("n_estimators", 2, 300),
        max_depth=trial.suggest_int("max_depth", 2, 15),
        random_state=666
    )


def create_xgboost_model(trial):
    return XGBRegressor(
        learning_rate=trial.suggest_uniform("learning_rate", 0.0000001, 2),
        n_estimators=trial.suggest_int("n_estimators", 2, 800),
        max_depth=trial.suggest_int("max_depth", 2, 20),
        gamma=trial.suggest_uniform('gamma', 0.0000001, 1),
        random_state=666
    )


def create_lgb_model(trial):
    return LGBMRegressor(
        learning_rate=trial.suggest_uniform('learning_rate', 0.0000001, 1),
        n_estimators=trial.suggest_int("n_estimators", 1, 800),
        max_depth=trial.suggest_int("max_depth", 2, 25),
        num_leaves=trial.suggest_int("num_leaves", 2, 3000),
        min_child_samples=trial.suggest_int('min_child_samples', 3, 200),
        random_state=666
    )


models = {
    'RFR': create_rf_model,
    'XGBR': create_xgboost_model,
    'LGB': create_lgb_model
}


def optimize(model_name, path, trials=30, sampler=TPESampler(seed=666), direction='maximize'):
    ds = pd.read_csv(path)
    X_ds, y_ds = ds.drop('count', axis=1), ds['count']
    X_train, X_val, y_train, y_val = train_test_split(X_ds, y_ds)

    def objective(trial):
        model = models[model_name](trial)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return rmsle(y_val, preds)

    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=trials)
    return study.best_params
