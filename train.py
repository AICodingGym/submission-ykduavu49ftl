"""
PA3 - Ventilator Pressure Prediction
MSE objective (faster convergence) + physics features + full breath context
Lung model: P ≈ R * flow + Volume/C
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import warnings, time
warnings.filterwarnings('ignore')

START = time.time()

print("Loading data...")
train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')
print(f"  train: {train.shape}, test: {test.shape}")

def add_features(df):
    df = df.copy().sort_values(['breath_id', 'time_step']).reset_index(drop=True)
    g = df.groupby('breath_id')

    df['time_delta']    = g['time_step'].diff().fillna(0)
    df['breath_len']    = g['time_step'].transform('count')
    df['pos_in_breath'] = g.cumcount() / (df['breath_len'] - 1).clip(lower=1)

    # PHYSICS: cumulative volume
    df['flow_dt']       = df['u_in'] * df['time_delta']
    df['u_in_volume']   = g['flow_dt'].cumsum()
    df['volume_remain'] = g['flow_dt'].transform('sum') - df['u_in_volume']
    df['u_in_total']    = g['u_in'].transform('sum')
    df['vol_frac']      = df['u_in_volume'] / (g['flow_dt'].transform('sum') + 1e-6)

    # PHYSICS: pressure components
    df['P_resist']      = df['R'] * df['u_in']
    df['P_compliance']  = df['u_in_volume'] / (df['C'] + 1e-6)
    df['P_model']       = df['P_resist'] + df['P_compliance']

    # Lag features (past context within breath)
    for lag in [1, 2, 3, 4, 5, 10, 15, 20, 30, 40]:
        df[f'u_in_lag{lag}']   = g['u_in'].shift(lag).fillna(0)
        df[f'P_model_lag{lag}'] = g['P_model'].shift(lag).fillna(0)

    # Lead features (future context — the FULL breath future)
    for lead in [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 79]:
        df[f'u_in_lead{lead}'] = g['u_in'].shift(-lead).fillna(0)

    # Derivatives
    df['u_in_diff1']    = g['u_in'].diff(1).fillna(0)
    df['u_in_diff2']    = g['u_in'].diff(2).fillna(0)
    df['P_model_diff1'] = g['P_model'].diff(1).fillna(0)

    # Rolling stats
    for w in [3, 5, 10, 20, 40]:
        df[f'u_in_roll{w}']     = g['u_in'].transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f'u_in_rollstd{w}']  = g['u_in'].transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))

    # Breath-level aggregates
    df['u_in_mean_b']    = g['u_in'].transform('mean')
    df['u_in_max_b']     = g['u_in'].transform('max')
    df['u_in_std_b']     = g['u_in'].transform('std').fillna(0)
    df['P_model_mean_b'] = g['P_model'].transform('mean')
    df['P_model_max_b']  = g['P_model'].transform('max')

    # Lung params
    df['RC']             = df['R'] * df['C']
    df['R_over_C']       = df['R'] / (df['C'] + 1e-6)
    df['u_out_lag1']     = g['u_out'].shift(1).fillna(0)
    df['u_out_lead1']    = g['u_out'].shift(-1).fillna(1)

    # Normalized physics
    df['P_resist_norm']     = df['P_resist'] / (df['P_model'].abs() + 1e-6)
    df['u_in_volume_norm']  = df['u_in_volume'] / (df['u_in_total'] + 1e-6)

    return df

print("Engineering features...")
train = add_features(train)
test  = add_features(test)

FEATURES = [
    'R', 'C', 'RC', 'R_over_C',
    'u_in', 'u_out', 'u_out_lag1', 'u_out_lead1',
    'P_resist', 'P_compliance', 'P_model',
    'P_model_lag1', 'P_model_lag2', 'P_model_lag3', 'P_model_lag4', 'P_model_lag5',
    'P_model_lag10', 'P_model_lag15', 'P_model_lag20', 'P_model_lag30', 'P_model_lag40',
    'P_model_diff1', 'P_model_mean_b', 'P_model_max_b',
    'P_resist_norm',
    'u_in_volume', 'volume_remain', 'u_in_total', 'flow_dt', 'vol_frac',
    'u_in_volume_norm',
    'u_in_lag1', 'u_in_lag2', 'u_in_lag3', 'u_in_lag4', 'u_in_lag5',
    'u_in_lag10', 'u_in_lag15', 'u_in_lag20', 'u_in_lag30', 'u_in_lag40',
    'u_in_lead1', 'u_in_lead2', 'u_in_lead3', 'u_in_lead4', 'u_in_lead5',
    'u_in_lead10', 'u_in_lead15', 'u_in_lead20', 'u_in_lead30',
    'u_in_lead40', 'u_in_lead50', 'u_in_lead60', 'u_in_lead79',
    'u_in_diff1', 'u_in_diff2',
    'u_in_roll3', 'u_in_roll5', 'u_in_roll10', 'u_in_roll20', 'u_in_roll40',
    'u_in_rollstd3', 'u_in_rollstd5', 'u_in_rollstd10',
    'u_in_mean_b', 'u_in_max_b', 'u_in_std_b',
    'time_step', 'time_delta', 'pos_in_breath', 'breath_len',
]

# Train on inspiratory rows only (only these are scored)
train_insp = train[train['u_out'] == 0].copy()
print(f"Training on inspiratory rows: {len(train_insp):,} / {len(train):,}")
print(f"Features: {len(FEATURES)}")

X_insp  = train_insp[FEATURES]
y_insp  = train_insp['pressure']
groups  = train_insp['breath_id']
X_test  = test[FEATURES]

N_FOLDS = 3
kf = GroupKFold(n_splits=N_FOLDS)

lgb_params = dict(
    objective        = 'mse',       # KEY CHANGE: MSE converges much faster than MAE
    num_leaves       = 1024,
    learning_rate    = 0.05,
    n_estimators     = 15000,
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
    bagging_freq     = 1,
    min_child_samples= 20,
    reg_alpha        = 0.05,
    reg_lambda       = 0.05,
    n_jobs           = -1,
    verbose          = -1,
)

oof_preds   = np.zeros(len(train_insp))
test_preds  = np.zeros(len(test))
fold_scores = []

print(f"\nTraining {N_FOLDS}-fold LightGBM (MSE objective, physics + full-breath features)...")
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_insp, y_insp, groups)):
    X_tr, X_val = X_insp.iloc[tr_idx], X_insp.iloc[val_idx]
    y_tr, y_val = y_insp.iloc[tr_idx], y_insp.iloc[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='mae',           # monitor MAE even though we train MSE
        callbacks=[
            lgb.early_stopping(300, verbose=False),
            lgb.log_evaluation(500),
        ],
    )

    val_pred           = model.predict(X_val)
    oof_preds[val_idx] = val_pred
    fold_mae           = mean_absolute_error(y_val, val_pred)
    fold_scores.append(fold_mae)
    test_preds        += model.predict(X_test) / N_FOLDS

    elapsed = time.time() - START
    print(f"  Fold {fold+1}/{N_FOLDS}  MAE={fold_mae:.5f}  best_iter={model.best_iteration_}  ({elapsed:.0f}s)")

oof_mae = mean_absolute_error(y_insp, oof_preds)
print(f"\nOOF inspiratory MAE: {oof_mae:.5f}  (mean folds: {np.mean(fold_scores):.5f})")

sub = test[['id']].copy()
sub['pressure'] = test_preds
sub.to_csv('submission.csv', index=False)
print(f"\nSaved submission.csv  ({len(sub)} rows)")
print(f"Total runtime: {(time.time()-START)/60:.1f} min")
print("\nNext: aicodinggym mle submit ventilator-pressure-prediction -F submission.csv")
