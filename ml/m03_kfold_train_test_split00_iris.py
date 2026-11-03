import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Kfold 실습(분류)
# 1. 데이터
x, y = load_iris(return_X_y=True)
print(x, y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8, 
    stratify=y,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

# 2. 모델 구성
model = DecisionTreeClassifier()

# 3, 4. 훈련, 평가
# 컴파일 없이 바로 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)

print("acc: ", scores,'\n평균 acc: ', round(np.mean(scores), 2))
# acc:  [1.         0.95833333 0.95833333 1.         0.95833333]         > 만약 값들이 비슷하면 좋은 데이터
# 평균 acc:  0.98

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)

print(y_predict)    # [2 0 2 1 1 1 1 0 2 0 0 0 0 2 2 2 0 1 0 1 2 0 1 1 2 0 1 1 1 1]
print(y_test)       # [2 0 2 1 1 1 2 0 2 0 0 0 0 2 2 2 0 2 0 1 2 0 1 1 2 0 1 1 1 1]

accuracy_score = accuracy_score(y_test, y_predict)
print("cross_val_predict ACC: ", accuracy_score)    # cross_val_predict ACC:  0.9333333333333333