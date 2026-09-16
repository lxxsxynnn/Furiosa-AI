import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd

# mnist 데이터 셋 확인해보기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train)
# print(x_train[0])
# print(x_train[0][0])

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

# y값 확인
print(np.unique(y_train, return_counts=True))   # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]))
print(pd.value_counts(y_test))                  # Name: count, dtype: int64

import matplotlib.pyplot as plt
plt.imshow(x_train[4133], 'gray')
plt.show()