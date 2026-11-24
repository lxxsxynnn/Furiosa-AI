import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor

# 트리 하나와 배깅·RandomForest 점수 비교 - 캘리포니아 주택 가격
# 1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=333,
    # stratify=y,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
# model = DecisionTreeRegressor()
# model = BaggingRegressor(
#     DecisionTreeRegressor(),
#     n_estimators=10,    # 모델을 몇 번 돌릴 것인지
#     random_state=333,
#     n_jobs=-1,
#     max_samples=0.8,
#     # bootstrap=True,     # Default(중복 허용)
#     bootstrap=False,  # 중복 허용 X
# )
model = RandomForestRegressor()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
result = model.score(x_test, y_test)
print('최종 점수: ', result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2: ', r2)

'''
[DecisionTreeRegressor]
최종 점수:  0.5931307159540253
r2:  0.5931307159540253

[BaggingRegressor + bootstrap=True]
최종 점수:  0.7709319970417915
r2:  0.7709319970417915

[BaggingRegressor + bootstrap=False]
최종 점수:  0.771260941840586
r2:  0.771260941840586

[RandomForestRegressor]
최종 점수:  0.802101096263111
r2:  0.802101096263111
'''