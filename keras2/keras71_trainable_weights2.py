import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

# 동결 여부에 따라 학습 결과 비교하기 (trainable = False면 fit을 해도 가중치가 안 바뀜)
# 실행할 때 같은 랜덤값이 나오게 고정하기
# random.seed(333)
# np.random.seed(333)
tf.random.set_seed(333)

# 1. 데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])

# 2. 모델 구성
model = Sequential()
model.add(Dense(30, input_dim=1))
model.add(Dense(15))
model.add(Dense(1))

################ 동결 ################
# model.trainable = False

# model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=2000, verbose=1)

# 4. 평가, 예측
y_predict = model.predict(x)
print(y_predict)