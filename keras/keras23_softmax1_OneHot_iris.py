import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.metrics import accuracy_score

# 다중 분류 모델 만들어보기
# 1. 데이터
datasets = load_iris()
# print(datasets)               # {'data' : ..., 'target' : ..., 'DESCR' : ..., ...}
print(datasets.DESCR)           # Summary Statistics에 Class Correlation > 클래스 간의 상관관계, 수치가 높을수록 서로 연관이 있다는 얘기
print(datasets.feature_names)   # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data
y = datasets['target']

print(x.shape, y.shape)                 # (150, 4) (150,)
print(y)
print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([50, 50, 50]))

'''
[0, 0, 1, 1, 2]         # (5,)
->
[[1, 0, 0]
[1, 0, 0]
[0, 1, 0]
[0, 1, 0]
[0, 0, 1]]              # (5, 3)
하나의 행렬 형태로 묶기
'''
# 벡터화 방법 1 - 텐서플로우
# OneHot 먼저 하고 split하는 게 작업 횟수가 적음
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)     # 라벨 값이 0으로 시작하는 게 아니면 빈 0이 들어간 컬럼이 생길 수 있음 주의

# 벡터화 방법 2 - pandas
y = pd.get_dummies(y, dtype=float).values

# 벡터화 방법 3 - sklearn
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder()                     # 2차원 배열을 받아야 하는데 1차원 배열이 왔음 > reshape 필요(reshape의 조건 1. 내용이 바뀌었는가? X, 2. 순서가 바뀌었는가? X)
# y = y.reshape(-1, 1)                      # 2차원 배열로 reshape
# ohe = OneHotEncoder(sparse_output=False)  # 기본적으로 sparse_output을 같이 출력해주는데 이걸 안쓰니까 sparse_output=False로 해야 함
# y = ohe.fit_transform(y)

print(y)
print(y.shape)  # (150, 3)

# exit()

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=100,
    # shuffle=False,    # 사용할 경우 최악(데이터가 00000...11111....2222 형태인데 섞지 않으면 특정 숫자에 대한 학습을 하고 다른 숫자에 대해 예측을 하게 됨)
    shuffle=True,
    stratify=y,
)

print(x_train.shape, x_test.shape)  # (120, 4) (30, 4)
print(y_train.shape, y_test.shape)  # (120, 3) (30, 3)

# exit()

# 2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim = 4, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))       # y 가공 없이 돌리면 에러 발생함("2는 1보다 크다", "1은 0과 2의 중간") > 각 라벨들의 가치를 동일하게 인식해야 함 → 데이터를 위치값으로 보자(벡터화)

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=100,
    restore_best_weights=True,
)

start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=4,
          verbose=1,
          validation_split=0.2,
          callbacks=[es],
          )
end_time = time.time()

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss : ', result[0])             # loss :  0.07946857064962387
print('acc : ', round(result[1], 2))    # acc :  0.97 소수점 길어서 반올림 처리

y_predict = model.predict(x_test)       # model.predict(x_test) : w * x_test + b

# 여기서도 [[1, 0, 0], [0, 0, 1], [0, 0, 1], ...] 이런 식으로 되어있는 걸 [0, 1, 2] 이렇게 변환해줘야 함
y_test = np.argmax(y_test, axis=1)
print(y_test)       # [1 1 0 0 2 0 1 2 1 1 1 0 0 2 0 2 2 1 2 0 2 1 1 2 0 2 2 0 0 1]
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)    # [1 1 0 0 2 0 1 2 1 1 1 0 0 2 0 2 2 1 2 0 2 1 1 2 0 2 2 0 0 2]

accuracy_score = accuracy_score(y_test, y_predict)  # 테스트 결과와 예측 결과를 비교해 얼마나 정확한가를 확인
print('acc_score : ', accuracy_score)       # acc_score :  0.9666666666666667
print('걸린 시간 : ', round(end_time - start_time, 2), '초')