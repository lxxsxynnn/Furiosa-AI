import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

# for문으로 파라미터 튜닝(분류)
# 1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
    stratify=y,
)

learning_rate = [0.1, 0.05, 0.01, 0.005, 0.001]
max_depth = [3, 4, 5, 6, 7]
best_score = 0
best_parameters = ''

for i, lr in enumerate(learning_rate):
    for j, md in enumerate(max_depth):  # tree 계열에서 아주 중요한 parameter
        # 2. 모델 구성
        model = HistGradientBoostingClassifier(
            learning_rate=lr, 
            max_depth=md,
        )

        # 3. 훈련
        model.fit(x_train, y_train)

        # 4. 평가
        score = model.score(x_test, y_test)

        if score > best_score:
            best_score = score
            best_parameters = lr, md
        print("Working on ", i, ", ", j, "th score: ", round(score, 4))

print("Best Score: {:.2f}".format(best_score))      # Best Score: 0.96
print("Best Parameters: ", best_parameters)         # Best Parameters:  (0.1, 5)