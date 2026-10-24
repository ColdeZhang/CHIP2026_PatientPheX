"""One fixed CPU estimator mapped from sklearn 1.8's own comparison helper."""

PARAMETERS = dict(
    boosting_type='gbdt', objective='binary', n_estimators=160,
    num_leaves=15, max_depth=-1, min_child_samples=35,
    learning_rate=.06, reg_lambda=5., reg_alpha=0.,
    max_bin=255, min_data_in_bin=1, min_child_weight=1e-3,
    min_split_gain=0., boost_from_average=True, enable_bundle=False,
    subsample_for_bin=200000, feature_fraction_bynode=1.,
    subsample=1., subsample_freq=0, colsample_bytree=1.,
    class_weight=None, is_unbalance=False, scale_pos_weight=1.,
    use_missing=True, zero_as_missing=False, feature_pre_filter=False,
    random_state=20261023, data_random_seed=20261023,
    feature_fraction_seed=20261023, bagging_seed=20261023,
    deterministic=True, force_col_wise=True, n_jobs=1,
    device_type='cpu', verbosity=-1,
)


def estimator():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**PARAMETERS)
