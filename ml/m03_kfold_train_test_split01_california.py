import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# Kfold 실습(회귀)
# 1. 데이터
x, y = fetch_california_housing(return_X_y=True)
print(x, y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=132, train_size=0.8,
)

n_split = 5
kFold = KFold(n_splits=n_split, shuffle=True, random_state=132)

# 2. 모델 구성
model = RandomForestRegressor()
# model = GradientBoostingRegressor()
# model = DecisionTreeRegressor()
# model = XGBRegressor()

# 3, 4. 훈련, 평가
scores = cross_val_score(model, x_train, y_train, cv=kFold)
y_predict = cross_val_predict(model, x_test, y_test, cv=kFold)

print("r2: ", scores, "\n평균 r2: ", round(np.mean(scores), 4))
r2_score = r2_score(y_test, y_predict)
print("cross_val_predict R2: ", r2_score)

'''
[RandomForest]
r2:  [0.80444277 0.81074339 0.7883809  0.80919714 0.79911738] 
평균 r2:  0.8024
cross_val_predict R2:  0.7597259139572329

[GradientBoosting]
r2:  [0.7899948  0.78440848 0.76494418 0.7924045  0.78443143] 
평균 r2:  0.7832
cross_val_predict R2:  0.7790445973280993

[DecisionTree]
r2:  [0.58589198 0.61128399 0.599773   0.57526391 0.60475515] 
평균 r2:  0.5954
cross_val_predict R2:  0.5265636387552994

[XGB]
r2:  [0.83454109 0.82962236 0.82397324 0.83989656 0.82441363] 
평균 r2:  0.8305
cross_val_predict R2:  0.7869573486774659
'''