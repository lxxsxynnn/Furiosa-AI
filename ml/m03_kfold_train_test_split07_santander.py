import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# Kfold 실습(회귀)
# 1. 데이터
data_path = "C:/study/_data/kaggle_santander/"

train_csv = pd.read_csv(data_path + "train.csv", index_col=0)
test_csv = pd.read_csv(data_path + "test.csv", index_col=0)

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=345, train_size=0.8,
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
print("r2: ", scores, "\n평균 r2: ", round(np.mean(scores), 4))

y_predict = cross_val_predict(model, x_test, y_test, cv=kFold)
r2_score = r2_score(y_test, y_predict)
print("cross_val_predict R2: ", r2_score)

'''
[RandomForest]
r2:  [0.13108924 0.1216116  0.13282316 0.13860098 0.13457321] 
평균 r2:  0.1317
cross_val_predict R2:  0.11155948810451399

[GradientBoosting]
r2:  [0.13493085 0.12913079 0.13658925 0.14002409 0.13476594] 
평균 r2:  0.1351
cross_val_predict R2:  0.13466607189163748

[DecisionTree]
r2:  [-0.83038834 -0.86359997 -0.83666128 -0.79340506 -0.80231676] 
평균 r2:  -0.8253
cross_val_predict R2:  -0.8591984708715479

[XGB]
r2:  [0.18999803 0.17848438 0.19303358 0.20863861 0.1992591 ] 
평균 r2:  0.1939
cross_val_predict R2:  0.12510037422180176
'''