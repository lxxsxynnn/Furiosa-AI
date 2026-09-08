from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt

# 과적합 눈으로 확인해보기 - 보스턴 주택 가격
# 1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()   #  tensorflow는 가져오면서 분할됨(훈련/학습 묶음으로 나오니까 주의할 것)
print(x_train.shape, x_test.shape)  # (404, 13) (102, 13)
print(y_train.shape, y_test.shape)  # (404,) (102,)

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True
)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=1,
          validation_split=0.25,
          callbacks=[es],
          )

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)      # loss : 22.92780113220215 (1000, 1)

# 시각화
plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'], c='yellow', label='loss')
plt.plot(hist.history['val_loss'], c='purple', label="val_loss")
plt.legend(loc="upper right")
plt.title("Boston Housing")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()