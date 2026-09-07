from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

# 과적합 눈으로 확인해보기 - 당뇨
# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=45)

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=10))
model.add(Dense(6))
model.add(Dense(6))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
hist = model.fit(x_train, y_train, epochs=640, batch_size=16,
          validation_split=0.33,
          )

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : " , loss)     # loss : 2383.88232421875 (0.75, 45, 640, 16)

# 그래프 그려보기
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'], c='red', label='loss')           # y값만 넣으면 시간순으로 그려줌
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.legend(loc='upper right')
plt.title('Diabetes Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()          # 격자표시 추가
plt.show()