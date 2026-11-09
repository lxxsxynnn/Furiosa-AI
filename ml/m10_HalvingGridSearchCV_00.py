from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection._search_successive_halving import HalvingGridSearchCV
from sklearn.experimental import enable_halving_search_cv
from xgboost import XGBClassifier
import time
import warnings
warnings.filterwarnings('ignore')

# HalvingGridSearchCV로 파라미터 튜닝(분류)
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
model = HalvingGridSearchCV(xgb, parameters, cv=kfold,     # 모델을 Wrapping한 알고리즘
                            verbose=1,
                        #    n_jobs=1,
                        #    random_state=333,
                        #    n_iter=10,       # 기본값. 42개 조합 중 10개만 뽑음
                            min_resources=50,  # 2 * cv * class 갯수
                            max_resources=1000, # 기본값 1437(x_train_set 개수)
                            factor=2,
                            refit=True,
                            )

# 3. 훈련
start = time.perf_counter()
model.fit(x_train, y_train)
end = time.perf_counter()

# print("최적의 매개변수: ", model.best_estimator_)
# print("최적의 파라미터: ", model.best_params_)
'''
n_iterations: 3
n_required_iterations: 4
n_possible_iterations: 3
min_resources_: 100
max_resources_: 1437
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 42
n_resources: 100
Fitting 5 folds for each of 42 candidates, totalling 210 fits
----------
iter: 1
n_candidates: 14
n_resources: 300
Fitting 5 folds for each of 14 candidates, totalling 70 fits
----------
iter: 2
n_candidates: 5
n_resources: 900
Fitting 5 folds for each of 5 candidates, totalling 25 fits
'''

# 4. 평가, 예측
# print("model.score: ", model.score(x_test, y_test))
print("best_score: ", model.best_score_)    # train data에 대한 결과 > 과적합
print("elapsed: ", round(end - start, 2), "sec")
'''
best_score:  0.9264369956548728
elapsed:  58.98 sec
'''

'''
n_iterations: 4
n_required_iterations: 6
n_possible_iterations: 4
min_resources_: 100
max_resources_: 1437
aggressive_elimination: False
factor: 2
----------
iter: 0
n_candidates: 42
n_resources: 100
Fitting 5 folds for each of 42 candidates, totalling 210 fits
----------
iter: 1
n_candidates: 21
n_resources: 200
Fitting 5 folds for each of 21 candidates, totalling 105 fits
----------
iter: 2
n_candidates: 11
n_resources: 400
Fitting 5 folds for each of 11 candidates, totalling 55 fits
----------
iter: 3
n_candidates: 6
n_resources: 800
Fitting 5 folds for each of 6 candidates, totalling 30 fits
best_score:  0.9410455974842767
elapsed:  84.41 sec
'''