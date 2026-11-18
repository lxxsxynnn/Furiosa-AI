import numpy as np
import matplotlib.pyplot as plt

# 활성화 함수 그려보기 - softmax(파이 차트)
# def softmax(x):
#     return np.exp(x) / np.sum(np.exp(x))

softmax = lambda x: np.exp(x) / np.sum(np.exp(x))

x = np.arange(1, 5)
y = softmax(x)

ratio = y
labels = y
plt.pie(ratio, labels=labels, shadow=True, startangle=90)
plt.show()