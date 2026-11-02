import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score #교차 검증
from sklearn.tree import DecisionTreeClassifier

# Kfold 실습(분류)
# 1. 데이터
x, y = load_iris(return_X_y=True)
print(x, y)

n_split = 5
kFold = KFold(n_splits=n_split, shuffle=True, random_state=333)

# 2. 모델 구성
model = DecisionTreeClassifier()

# 3, 4. 훈련, 평가
# 컴파일 없이 바로 훈련
scores = cross_val_score(model, x, y, cv=kFold)


# print("acc: ", scores,)     # acc:  [0.93333333 0.9        1.         0.96666667 0.93333333]
print("acc: ", scores,'\n평균 acc: ', round(np.mean(scores), 2))
# acc:  [0.93333333 0.9        1.         0.96666667 0.93333333]        > 만약 값들이 비슷하면 좋은 데이터
# 평균 acc:  0.9