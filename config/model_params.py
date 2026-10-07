from scipy.stats import randint, uniform

# Hyperparameter distribution for LGBMClassifier inside Scikit-Learn Pipeline
LIGHTGBM_PIPELINE_PARAMS = {
    'classifier__n_estimators': randint(100, 500),
    'classifier__max_depth': randint(5, 50),
    'classifier__learning_rate': uniform(0.01, 0.2),
    'classifier__num_leaves': randint(20, 100),
    'classifier__boosting_type': ['gbdt', 'dart', 'goss']
}

LIGHTGM_PARAMS = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(5, 50),
    'learning_rate': uniform(0.01, 0.2),
    'num_leaves': randint(20, 100),
    'boosting_type': ['gbdt', 'dart', 'goss']
}

RANDOM_SEARCH_PARAMS = {
    'n_iter': 3,
    'cv': 3,
    'n_jobs': -1,
    'verbose': 2,
    'random_state': 42,
    'scoring': 'f1'
}