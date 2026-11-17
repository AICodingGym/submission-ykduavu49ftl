"""
PA3 - Ventilator Pressure Prediction
LightGBM - eval on inspiratory rows only, more features, proper convergence
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

    # Lag features (past values within breath)
    for lag in range(1, 6):
        df[f'u_in_lag{lag}'] = g['u_in'].shift(lag).fillna(0)

    # Lead features (FUTURE values — we have full breath sequences in both train & test!)
    for lead in range(1, 6):
        df[f'u_in_lead{lead}'] = g['u_in'].shift(-lead).fillna(0)

    # Derivatives
    df['u_in_diff1'] = g['u_in'].diff(1).fillna(0)
    df['u_in_diff2'] = g['u_in'].diff(2).fillna(0)

    # Cumulative u_in (work done so far)
    df['u_in_cumsum'] = g['u_in'].cumsum()

    # Remaining u_in work (total breath u_in - cumsum so far)
    df['u_in_total']   = g['u_in'].transform('sum')
    df['u_in_remain']  = df['u_in_total'] - df['u_in_cumsum']

    # Rolling stats
    df['u_in_roll3']  = g['u_in'].transform(lambda x: x.rolling(3,  min_periods=1).mean())
    df['u_in_roll5']  = g['u_in'].transform(lambda x: x.rolling(5,  min_periods=1).mean())
    df['u_in_roll10'] = g['u_in'].transform(lambda x: x.rolling(10, min_periods=1).mean())

    # Time features
    df['time_delta']  = g['time_step'].diff().fillna(0)
    df['breath_len']  = g['time_step'].transform('count')
    df['pos_in_breath'] = g.cumcount() / (df['breath_len'] - 1).clip(lower=1)

    # Lung parameter interactions
    df['RC']           = df['R'] * df['C']
    df['R_over_C']     = df['R'] / (df['C'] + 1e-6)
    df['u_in_x_R']     = df['u_in'] * df['R']
    df['u_in_x_C']     = df['u_in'] * df['C']
    df['u_in_x_RC']    = df['u_in'] * df['RC']
    df['cumsum_x_R']   = df['u_in_cumsum'] * df['R']
    df['cumsum_x_C']   = df['u_in_cumsum'] * df['C']

    # u_out transitions
    df['u_out_lag1']   = g['u_out'].shift(1).fillna(0)
    df['u_out_diff']   = df['u_out'] - df['u_out_lag1']

    # Breath-level stats
    df['u_in_mean_b']  = g['u_in'].transform('mean')
    df['u_in_max_b']   = g['u_in'].transform('max')
    df['u_in_std_b']   = g['u_in'].transform('std').fillna(0)
    df['u_in_last']    = g['u_in'].transform('last')
    df['u_in_first']   = g['u_in'].transform('first')

    return df

print("Engineering features...")
train = add_features(train)
test  = add_features(test)

FEATURES = [
    'R', 'C', 'RC', 'R_over_C',
    'u_in', 'u_out',
    'u_in_lag1', 'u_in_lag2', 'u_in_lag3', 'u_in_lag4', 'u_in_lag5',
    'u_in_lead1', 'u_in_lead2', 'u_in_lead3', 'u_in_lead4', 'u_in_lead5',
    'u_in_diff1', 'u_in_diff2',
    'u_in_cumsum', 'u_in_remain', 'u_in_total',
    'u_in_roll3', 'u_in_roll5', 'u_in_roll10',
    'u_in_x_R', 'u_in_x_C', 'u_in_x_RC',
    'cumsum_x_R', 'cumsum_x_C',
    'u_in_mean_b', 'u_in_max_b', 'u_in_std_b', 'u_in_last', 'u_in_first',
    'u_out_lag1', 'u_out_diff',
    'time_step', 'time_delta',
    'pos_in_breath', 'breath_len',
]

X      = train[FEATURES]
y      = train['pressure']
X_test = test[FEATURES]
groups = train['breath_id']

N_FOLDS = 3
kf = GroupKFold(n_splits=N_FOLDS)

lgb_params = dict(
    objective        = 'mae',
    num_leaves       = 512,
    learning_rate    = 0.05,
    n_estimators     = 8000,       # plenty of room; early stopping will cut early
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
    bagging_freq     = 1,
    min_child_samples= 20,
    reg_alpha        = 0.05,
    reg_lambda       = 0.05,
    n_jobs           = -1,
    verbose          = -1,
)

oof_preds   = np.zeros(len(train))
test_preds  = np.zeros(len(test))
fold_scores = []

print(f"\nTraining {N_FOLDS}-fold LightGBM (eval on inspiratory rows only)...")
for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y, groups)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    # KEY: early stop using ONLY inspiratory rows so we optimise the right metric
    insp_mask = train.iloc[val_idx]['u_out'] == 0
    X_val_insp = X_val[insp_mask]
    y_val_insp = y_val[insp_mask]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val_insp, y_val_insp)],
        callbacks=[
            lgb.early_stopping(200, verbose=False),
            lgb.log_evaluation(500),
        ],
    )

    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred

    fold_mae = mean_absolute_error(y_val_insp, val_pred[insp_mask])
    fold_scores.append(fold_mae)

    test_preds += model.predict(X_test) / N_FOLDS
    elapsed = time.time() - START
    best_it = model.best_iteration_
    print(f"  Fold {fold+1}/{N_FOLDS}  insp-MAE={fold_mae:.5f}  best_iter={best_it}  ({elapsed:.0f}s elapsed)")

insp_mask_all = train['u_out'] == 0
oof_mae = mean_absolute_error(y[insp_mask_all], oof_preds[insp_mask_all])
print(f"\nOOF inspiratory MAE: {oof_mae:.5f}  (mean folds: {np.mean(fold_scores):.5f})")
print(f"Expected leaderboard score: ~{oof_mae:.4f}")

sub = test[['id']].copy()
sub['pressure'] = test_preds
sub.to_csv('submission.csv', index=False)
print(f"\nSaved submission.csv  ({len(sub)} rows)")
print(f"Total runtime: {(time.time()-START)/60:.1f} min")
print("\nNext: aicodinggym mle submit ventilator-pressure-prediction -F submission.csv")
