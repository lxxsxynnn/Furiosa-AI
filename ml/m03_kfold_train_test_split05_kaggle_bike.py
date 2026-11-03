import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# Kfold 실습(회귀)
# 1. 데이터
path = "C:/study/_data/kaggle_bike/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)    # 날짜 시간 데이터를 인덱스로
test_csv = pd.read_csv(path + "test.csv", index_col=0)
train_csv = train_csv.drop(['casual', 'registered'], axis=1)   # test에 없는 컬럼 제거

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']  # train_csv에서 count만 사용

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=234, train_size=0.8,
)

n_split = 5
kFold = KFold(n_splits=n_split, shuffle=True, random_state=132)

# 2. 모델 구성
# model = RandomForestRegressor()
# model = GradientBoostingRegressor()
# model = DecisionTreeRegressor()
model = XGBRegressor()

# 3, 4. 훈련, 평가
scores = cross_val_score(model, x_train, y_train, cv=kFold)
print("r2: ", scores, "\n평균 r2: ", round(np.mean(scores), 4))

y_predict = cross_val_predict(model, x_test, y_test, cv=kFold)
r2_score = r2_score(y_test, y_predict)
print("cross_val_predict R2: ", r2_score)

'''
[RandomForest]
r2:  [0.29906309 0.3160996  0.24995476 0.31604701 0.28716165] 
평균 r2:  0.2937
cross_val_predict R2:  0.269943040812909

[GradientBoosting]
r2:  [0.34240136 0.34201205 0.29996869 0.34346436 0.31261698] 
평균 r2:  0.3281
cross_val_predict R2:  0.31107701303563806

[DecisionTree]
r2:  [-0.18035506 -0.13152605 -0.19689021 -0.10854902 -0.24006949] 
평균 r2:  -0.1715
cross_val_predict R2:  -0.2112703724472198

[XGB]
r2:  [0.34053463 0.32536113 0.27071446 0.33815849 0.28482491] 
평균 r2:  0.3119
cross_val_predict R2:  0.19500070810317993
'''