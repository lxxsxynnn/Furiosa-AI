import numpy as np
import matplotlib.pyplot as plt

# 활성화 함수 그려보기 - tanh
x = np.arange(-5, 5, 0.1)
y = np.tanh(x)

plt.plot(x, y)
plt.grid()
plt.show()