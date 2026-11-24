import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

# Stacking - 당뇨
# 1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=333,
    # stratify=y,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2-1. 모델 구성
xgb = XGBRegressor()
lgbm = LGBMRegressor(verbosity=-1)
cat = CatBoostRegressor(verbose=0)
rf = RandomForestRegressor()

models = [xgb, lgbm, cat, rf]
train_list = []
test_list = []

for model in models:
    # 3-1. 훈련
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_list.append(y_pred)
    test_list.append(y_test_pred)

    score = r2_score(y_test, y_test_pred)
    class_name = model.__class__.__name__
    print('{0} R2: {1:.4f}'.format(class_name, score))

'''
XGBRegressor R2: 0.2515
LGBMRegressor R2: 0.3441
CatBoostRegressor R2: 0.3786
RandomForestRegressor R2: 0.3852
'''

x_train_new = np.array(train_list).T
print(x_train_new.shape)    # (353, 4)

x_test_new = np.array(test_list).T
print(x_test_new.shape)     # (89, 4)

# 2-2. 모델 구성
model2 = CatBoostRegressor(verbose=0)

# 3-2. 훈련
model2.fit(x_train_new, y_train)

# 4. 평가, 예측
y_pred_new = model2.predict(x_test_new)
score_new = r2_score(y_test, y_pred_new)
print('Stacking Result: ', score_new)
# Stacking Result:  0.3198020164183043