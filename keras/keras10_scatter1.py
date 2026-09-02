import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 데이터 시각화 해보기
# 1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 7, 5, 7, 8, 6, 10])

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                  # test_size=0.3,
                                                  # shuffle=True,
                                                    random_state=123,
                                                    )

print('x_train : ', x_train)
print('x_test : ', x_test)
print('y_train : ', y_train)
print('y_test : ', y_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim = 1))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
model.fit(x_train, y_train, epochs=500, batch_size=1)

print("=======================================================")
# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)     # evaluate는 최종 결과를 가지고 1번 수행, batch_size(default=32) 명시하면 나눠서 수행(w갱신은 없음)
print("loss: ", loss)           # loss:  1.4086682043057408e-08
results = model.predict(np.array(x))
print("results : ", results)    # results :  [[10.999918]]

# 그래프 그리기
import matplotlib.pyplot as plt

plt.scatter(x, y)                   # 데이터 점 찍기
plt.plot(x, results, color="red")   # 선으로 잇기
plt.show()