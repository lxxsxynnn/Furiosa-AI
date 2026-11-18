import numpy as np
import matplotlib.pyplot as plt

# 활성화 함수 그려보기 - sigmoid
# def simoid(x):
#     return 1 / (1 + np.exp(-x))

sigmoid = lambda x: 1 / (1+ np.exp(-x))

x= np.arange(-5, 5, 0.1)
print(x)
print(len(x))   # 100

y = sigmoid(x)

plt.plot(x, y)
plt.grid()
plt.show()