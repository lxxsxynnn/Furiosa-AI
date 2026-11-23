from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 논리 게이트를 직선 하나로 가르는 모델로 풀어 보기 - XOR
# 1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 0]

# 2. 모델 구성
model = Perceptron()
# model = LinearSVC()

# 3. 훈련
model.fit(x_data, y_data)

# 4. 평가, 예측
y_predict = model.predict(x_data)

result = model.score(x_data, y_data)
print("model.score: ", result)  # model.score:  0.5

acc = accuracy_score(y_data, y_predict)
print("accuracy score: ", acc)  # accuracy score:  0.5
# LinearSVC도 0.5. 둘 다 네 점을 모두 0으로 찍음