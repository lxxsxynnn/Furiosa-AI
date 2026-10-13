import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

# 데이터 1개로 학습한 뒤 가중치를 꺼내 직접 계산해 보기
# 실행할 때 같은 랜덤값이 나오게 고정하기
random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

# 1. 데이터
x = np.array([1])
y = np.array([1])

# 2. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

################ 동결 ################
# model.trainable = False

# model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=1000, verbose=1)

# 4. 평가, 예측
y_predict = model.predict(x)
print(y_predict)

print(model.weights)


# [<Variable path=sequential/dense/kernel, shape=(1, 3), dtype=float32, value=[[ 0.7488525  -0.70411927 -1.0218748 ]]>, 
# <Variable path=sequential/dense/bias, shape=(3,), dtype=float32, value=[-0.10278901 -0.17491737 -0.11062892]>,
'''
h1 = 1 * 0.7488525 + (-0.10278901) =  0.64606349
h2  = 1 * -0.70411927 + (-0.17491737) = -0.87903664
h3 = 1 *  -1.0218748 + (-0.11062892) = -1.13250372
'''

#  <Variable path=sequential/dense_1/kernel, shape=(3, 2), dtype=float32, value=[[ 0.5455212  -0.9855041 ] [-0.2744968  -0.33626312] [ 0.30810106 -0.9928022 ]]>, 
#  <Variable path=sequential/dense_1/bias, shape=(2,), dtype=float32, value=[-0.09878281  0.10759086]>, 
'''
o1 = h1 * 0.5455212 + h2 * -0.2744968 + h3 * 0.30810106 + (-0.09878281)
=  0.64606349 * 0.5455212 + -0.87903664 * -0.2744968 + -1.13250372 * 0.30810106 + (-0.09878281)
= 0.1460256685177968

o2 = h1 * -0.9855041 + h2 * -0.33626312 + h3 * -0.9928022 + 0.10759086
= 0.64606349 * -0.9855041 + -0.87903664 * -0.33626312 + -1.13250372 * -0.9928022 + 0.10759086
= 0.8908324296295918
'''

#  <Variable path=sequential/dense_2/kernel, shape=(2, 1), dtype=float32, value=[[-0.42511365] [ 1.0762275 ]]>, 
#  <Variable path=sequential/dense_2/bias, shape=(1,), dtype=float32, value=[0.10333957]>]
'''
y = o1* -0.42511365 + o2 * 1.0762275 + 0.10333957
= 0.1460256685177968 * -0.42511365 + 0.8908324296295918 * 1.0762275 + 0.10333957
= 1.00000042372189082132818
'''