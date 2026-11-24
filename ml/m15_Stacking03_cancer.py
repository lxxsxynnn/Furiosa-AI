import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

# Stacking - 유방암
# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=333,
    stratify=y,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2-1. 모델 구성
xgb = XGBClassifier()
lgbm = LGBMClassifier(verbosity=-1)
cat = CatBoostClassifier(verbose=0)
rf = RandomForestClassifier()

model = StackingClassifier(
    estimators=[('XGB', xgb), ('LG', lgbm), ('CAT', cat), ('RF', rf)],
    final_estimator=CatBoostClassifier(verbose=0),
    n_jobs=-1,
    cv=5,   # kfold 지정하지 않아도 됨
)

# 3-2. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
y_pred = model.predict(x_test)
score_new = accuracy_score(y_test, y_pred)
print('Stacking Result: ', score_new)
# Stacking Result:  0.9912280701754386