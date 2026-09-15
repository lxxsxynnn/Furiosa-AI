import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer     # 유방암 관련 데이터셋 불러오기(이진분류, y가 0/1)

# 최고 가중치 불러오기 - 유방암
# 1. 데이터
datasets = load_breast_cancer()     # sklearn에서 제공하는 교육용 데이터셋, 실무에서 쓸 일이 없음
x = datasets['data']
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=234,
    stratify=y,         # 범주형 데이터에서는 불균형하게 나눠지는 걸 막기 위한 기능이 있음 / y를 기준으로 동일한 비율로 데이터를 잘라줌
)

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델링
save_path = 'C:/study/_save/keras31/'
model = load_model(save_path + 'k31_0914_1428_06_cancer_0005-0.0840.keras')

# 3. 컴파일, 훈련

# # 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('acc : ', round(loss[1], 4))

y_predict = model.predict(x_test)
y_predict = np.round(y_predict)

acc_score = accuracy_score(y_test, y_predict)

print('acc_score : ', acc_score)

'''
loss :  0.07397156208753586
acc :  0.9766
acc_score :  0.9766081871345029
'''