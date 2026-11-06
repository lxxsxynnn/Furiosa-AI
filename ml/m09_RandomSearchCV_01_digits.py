from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import time
import warnings
warnings.filterwarnings('ignore')

# RandomizedSearchCV로 파라미터 튜닝(분류)
# 1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
    stratify=y,
)

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333,)

parameters = [
    {'n_estimators': [100, 500], 'max_depth': [2, 4, 6],
     'learning_rate': [0.1, 0.15, 0.01, 0.005, 0.001]},  # 30
    {'max_depth': [6, 8, 10, 12], 'learning_rate': [0.1, 0.15, 0.01, 0.005, 0.001]},  # 20
    {'min_child_weight': [2, 3, 5, 7], 'learning_rate': [0.1, 0.15, 0.01, 0.005, 0.001]},    # 20
]   # 70

# 2. 모델
xgb = XGBClassifier()
model = RandomizedSearchCV(xgb, parameters, cv=kfold,     # 모델을 Wrapping한 알고리즘
                           verbose=1,
                           n_jobs=1,
                           random_state=333,
                           n_iter=10,       # 기본값. 70개 조합 중 10개만 뽑음
                           refit=False,
                           )

# 3. 훈련
start = time.perf_counter()
model.fit(x_train, y_train)
end = time.perf_counter()

# print("최적의 매개변수: ", model.best_estimator_)
# print("최적의 파라미터: ", model.best_params_)
'''
[refit 기본, random_state 없음]
최적의 매개변수:  XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              feature_weights=None, gamma=None, grow_policy=None,
              importance_type=None, interaction_constraints=None,
              learning_rate=0.15, max_bin=None, max_cat_threshold=None,
              max_cat_to_onehot=None, max_delta_step=None, max_depth=6,
              max_leaves=None, min_child_weight=None, missing=nan,
              monotone_constraints=None, multi_strategy=None, n_estimators=500,
              n_jobs=None, num_parallel_tree=None, ...)
최적의 파라미터:  {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.15}
'''

# 4. 평가, 예측
# print("model.score: ", model.score(x_test, y_test))
print("best_score: ", model.best_score_)    # train data에 대한 결과 > 과적합
print("elapsed: ", round(end - start, 2), "sec")
'''
[refit 기본, random_state 없음]
model.score:  0.9472222222222222
best_score:  0.9624249903213318
elapsed:  33.5 sec
'''

'''
[random_state, refit 적용 후]
Fitting 5 folds for each of 10 candidates, totalling 50 fits
best_score:  0.9645059039876113
elapsed:  17.05 sec
'''