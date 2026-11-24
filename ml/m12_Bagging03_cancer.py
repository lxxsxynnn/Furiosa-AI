import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

# 트리 하나와 배깅·RandomForest 점수 비교 - 유방암
# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=333,
    stratify=y,
)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# 2. 모델 구성
# model = DecisionTreeClassifier()
# model = BaggingClassifier(
#     DecisionTreeClassifier(),
#     n_estimators=10,    # 모델을 몇 번 돌릴 것인지
#     random_state=42,
#     n_jobs=-1,
#     max_samples=0.8,
#     # bootstrap=True,     # Default(중복 허용)
#     bootstrap=False,  # 중복 허용 X
# )
model = RandomForestClassifier()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
result = model.score(x_test, y_test)
print('최종 점수: ', result)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('acc: ', acc)

'''
[DecisionTreeClassifier]
최종 점수:  0.8947368421052632
acc:  0.8947368421052632

[BaggingClassifier + bootstrap=True]
최종 점수:  0.9736842105263158
acc:  0.9736842105263158

[BaggingClassifier + bootstrap=False]
최종 점수:  0.9649122807017544
acc:  0.9649122807017544

[RandomForestClassifier]
최종 점수:  0.9824561403508771
acc:  0.9824561403508771
'''