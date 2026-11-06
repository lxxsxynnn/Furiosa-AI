import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import time
import warnings
warnings.filterwarnings('ignore')

# GridSearchCV로 파라미터 튜닝(분류)
# 1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
    stratify=y,
)

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333,)

parameters = [
    {'n_estimators': [100, 500], 'max_depth': [6, 10, 12],
     'learning_rate': [0.1, 0.01, 0.001]},  # 18
    {'max_depth': [6, 8, 10, 12], 'learning_rate': [0.1, 0.01, 0.001]},  # 12
    {'min_child_weight': [2, 3, 5, 10], 'learning_rate': [0.1, 0.01, 0.001]},    # 12
]   # 42

# 2. 모델
xgb = XGBClassifier()
model = GridSearchCV(xgb, parameters, cv=kfold,     # 모델을 Wrapping한 알고리즘
                     verbose=1,
                     n_jobs=1,
                     refit=True,
                    #  random_state=333,    전부 다 돌리기 때문에 random_state(어떤 파라미터를 선택할 것인가에 대한 난수)가 필요없음
                    )

# 3. 훈련
start = time.perf_counter()
model.fit(x_train, y_train)
end = time.perf_counter()

print("최적의 매개변수: ", model.best_estimator_)
print("최적의 파라미터: ", model.best_params_)
'''
최적의 매개변수:  XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              feature_weights=None, gamma=None, grow_policy=None,
              importance_type=None, interaction_constraints=None,
              learning_rate=0.15, max_bin=None, max_cat_threshold=None,
              max_cat_to_onehot=None, max_delta_step=None, max_depth=2,
              max_leaves=None, min_child_weight=None, missing=nan,
              monotone_constraints=None, multi_strategy=None, n_estimators=500,
              n_jobs=None, num_parallel_tree=None, ...)
최적의 파라미터:  {'learning_rate': 0.15, 'max_depth': 2, 'n_estimators': 500}
'''

# 4. 평가, 예측
print("model.score: ", model.score(x_test, y_test))
print("best_score: ", model.best_score_)    # train data에 대한 결과 > 과적합
print("elapsed: ", round(end - start, 2), "sec")
'''
model.score:  0.9583333333333334
best_score:  0.968684668989547
elapsed:  161.54 sec
'''