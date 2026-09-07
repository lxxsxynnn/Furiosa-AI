import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# verbose(출력(로그) 옵션. 학습 결과에는 영향 없음)
# 1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

x_train = np.array([1, 2, 3, 4, 5, 6, 7])
y_train = np.array([1, 2, 3, 4, 5, 6, 7])

x_test = np.array([8, 9, 10])
y_test = np.array([8, 9, 10])

# 2. 모델
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(6))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
model.fit(x_train, y_train, epochs=100, batch_size=1,
          verbose=0,
         )

# verbose=0 : 학습 과정 안보여줌
# verbose=1 : 디폴트                           (Epoch 89/100    7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0051 )
# verbose=2 : 진행바 사라짐                     (Epoch 65/100    7/7 - 0s - 4ms/step - loss: 0.0080)
# verbose=그외 : 몇 번째 epoch 실행 중인지만 나옴 (Epoch 98/100)

# verbose를 바꾸는 이유 : 속도가 아니라 로그 가독성 때문
#   (실측 결과 verbose=0과 1의 학습 시간 차이는 1~3% 수준)
#   0 : 반복 실험 - 로그에 결과가 묻히는 걸 막음
#   1 : 지켜볼 때 - val_loss 추이, 발산/NaN 조기 발견
#   2 : 파일로 남길 때 - 진행바(\r)가 파일에서 지저분해지므로

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)                  # loss :  0.0071001313626766205