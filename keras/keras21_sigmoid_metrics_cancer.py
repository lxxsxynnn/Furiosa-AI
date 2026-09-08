import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_breast_cancer     # 유방암 관련 데이터셋 불러오기(범주형 데이터)

# 1. 데이터
datasets = load_breast_cancer()     # sklearn에서 제공하는 교육용 데이터셋, 실무에서 쓸 일이 없음

# print(datasets.DESCR)   # :Number of Instances: 569, :Number of Attributes: 30 numeric, predictive attributes and the class -> (569, 30) , 각 속성에 대한 상세 정보 등 제공 / pandas에서는 describe
print(datasets.feature_names)   # pandas에서는 columns

# x = datasets.data
x = datasets['data']
# 두 방법 모두 사용 가능
# .data > 원래 딕셔너리에서 값을 가져올 때는 .${key_name} 형태로 가져와야 함
# ['data']
# sklearn에서 wrapping한 데이터라 직접 접근 불가하고 위 방법으로 접근 가능
y = datasets.target

print(x.shape, y.shape)                     # (569, 30) (569,)
# type - 파이썬 함수
print(type(x))                              # <class 'numpy.ndarray'> > pandas 자체도 numpy로 구성되어 있음

print(y)                                    # 분류형 모델에서 y 범주 항상 확인할 것
print(np.unique(y))                         # y라벨이 뭔지 확인하고 싶을 때 / numpy에서 제공
# 각 라벨별 데이터가 몇개씩 있는지 알아보자
print(np.unique(y, return_counts=True))     # (array([0, 1]), array([212, 357])) > 범주형 데이터에서는 각 라벨 내 데이터 갯수가 비슷할수록 성능이 좋아짐

print(pd.DataFrame(y).value_counts())        # pandas에서 각 라벨별 데이터 갯수 세는 방법 cf) pandas는 DataFrame, Series만 존재
print(pd.Series(y).value_counts())
'''
1    357
0    212
Name: count, dtype: int64
'''

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.66, random_state=234,
# )

# print(np.unique(y_train, return_counts=True))   # (array([0, 1]), array([141, 234]))
# print(np.unique(y_test, return_counts=True))    # (array([0, 1]), array([ 71, 123]))
# 평가 데이터는 불균형해도 상관 없는데, 학습 데이터는 한쪽으로 쏠려있으면 좋지 않음
# 처음부터 트레인 데이터를 골고루 나눠주면 더 좋지 않을까?

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=234,
    stratify=y,         # 범주형 데이터에서는 불균형하게 나눠지는 걸 막기 위한 기능이 있음 / y를 기준으로 동일한 비율로 데이터를 잘라줌
)

print(np.unique(y_train, return_counts=True))   # (array([0, 1]), array([148, 250]))
print(np.unique(y_test, return_counts=True))    # (array([0, 1]), array([ 64, 107]))

print(x_train.shape, x_test.shape)              # (398, 30) (171, 30)
print(y_train.shape, y_test.shape)              # (398,) (171,)

# 2. 모델링
model = Sequential()
model.add(Dense(32, input_dim = 30, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(72, activation='relu'))
model.add(Dense(36, activation='relu'))
model.add(Dense(1, activation='sigmoid'))   # 마지막은 무조건 sigmoid

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',     # loss는 w 갱신할 때 사용하는데 이진분류는 무조건 binary_crossentropy
            #   metrics=['accuracy'],
              metrics=['acc'],      # train data 지표
             )  

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=20,
                   restore_best_weights=True,
                  )

start_time = time.time()

model.fit(x_train, y_train, 
          epochs=1000,
          batch_size=32,
          verbose=1,
          validation_split=0.3,
          callbacks=[es],
         )

end_time = time.time()

# # 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
# print("loss : ", loss)  # loss :  [0.21258428692817688, 0.9356725215911865] > 뒤에 오는 게 accuracy
print("==============================")
print('loss : ', loss[0])           # loss :  0.21817214787006378
print('acc : ', round(loss[1], 4))  # acc :  0.9298

y_predict = model.predict(x_test)
# print(y_predict[:10])

# Classification metrics can't handle a mix of binary and continuous targets
# y_test의 값은 0 또는 1로 되어있는데 y_predict는 0에서 1사이의 값으로 되어있어서 에러 발생(이진값과 연속적인 값을 함께 처리할 수 없음)
# 연속적인 값을 반올림 처리를 해서 해결

y_predict = np.round(y_predict)
print(y_predict[:10])

acc_score = accuracy_score(y_test, y_predict)

print('acc_score : ', acc_score)        # acc_score :  0.935672514619883

print("걸린 시간: ", round(end_time - start_time, 2), " 초")    # 걸린 시간:  8.49  초
