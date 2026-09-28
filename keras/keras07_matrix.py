import numpy as np

# 데이터의 shape에 익숙해지기
x1 = np.array([1, 2, 3])
print("x1 = ", x1.shape)    # x1 =  (3,)

# 데이터를 받으면 데이터 구조 먼저 확인
# 추후에 모델 구성할 때 사용
# model.add(Dense(3, input_dim = 1))
# 여기서 dim > dimension의 약자

x2 = np.array([[1, 2, 3]])
print("x2 =", x2.shape)     # x2 = (1, 3)

x3 = np.array([[1, 2], [3, 4]])
print("x3 = ", x3.shape)    # x3 = (2, 2)

x4 = np.array([[1, 2], [3, 4], [5, 6]])
print("x4 = ", x4.shape)    # x4 = (3, 2)

# x5 = np.array([[1, 2], [3, 4], [5, 6, 7]]) # 이런 유형의 데이터는 존재할 수 없음
# x5 = np.array([[1, 2], [3, 4], [5, 6,]])   # 콤마(,)로 끝나는 건 상관없음 -> ,는 뒤에 쓰고 싶은 거 써도 상관없다는 의미
x5 = np.array([[[1, 2], [3, 4], [5, 6,]]])
print("x5 = ", x5.shape)    # x5 = (1, 3, 2)

x6 = np.array([[[1, 2,], [3, 4]], [[5, 6,], [7, 8]]])
print("x6 = ", x6.shape)    # x6 = (2, 2, 2)

x7 = np.array([[[[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]]]])
print("x7 = ", x7.shape)    # x7 = (1, 1, 1, 2, 5)

x8 = np.array([[[1, 2, 3]], [[4, 5, 6]]])
print("x8 = ", x8.shape)    # x8 = (2, 1, 3)

x9 = np.array([[[[1]]], [[[2]]]])
print("x9 = ", x9.shape)    # x9 = (2, 1, 1, 1)
