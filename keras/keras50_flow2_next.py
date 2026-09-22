from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

# 데이터 증폭하기
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

data_gen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,   # 수평 뒤집기
    # vertical_flip=True,     # 수직 뒤집기
    width_shift_range=0.1,   # 평행이동
    height_shift_range=0.1,  # 평행이동
    rotation_range=5,       # 각도 조절(입력한 값만큼 이미지 회전)
    # zoom_range=0.2,
    shear_range=0.7,        # 좌표 하나를 고정하고 다른 몇개의 좌표를 이동
    fill_mode='nearest',    # 데이터를 옮기면 빈 공간이 생기게 되는데 그 근처 값으로 채우겠다는 의미
)

augment_size = 100

print(x_train.shape)    # (60000, 28, 28)
print(x_train[0].shape) # (28, 28)

# aaa = np.tile(x_train[0], augment_size)
# print(aaa.shape)    # (28, 2800)

# 데이터 단순 복붙
# 주의) shape은 맞지만 이미지가 깨진다
# np.tile은 가로로 이어붙이므로 (28, 2800)의 0번 행이 [행0][행0]...100개가 됨
# 여기서 784개씩 끊으면 첫 장이 '행0을 28번 반복한 가로줄무늬'가 됨
aaa = np.tile(x_train[0], augment_size).reshape(-1, 28, 28, 1)
print(aaa.shape)    # (100, 28, 28, 1)

xy = data_gen.flow(
                    # 한 줄(784)로 편 뒤에 tile해야 [이미지][이미지]...가 되어 온전한 100장이 나옴
                    np.tile(x_train[0].reshape(28 * 28), augment_size).reshape(-1, 28, 28, 1),
                    np.zeros(augment_size),     # 100개의 데이터에 0을 채워넣음
                    batch_size=augment_size,
                    shuffle=False,
).next()    # flow로 나오는 데이터는 Iterator 형태이기 때문에 next()를 해야 값을 꺼내 수 있음

print(xy)
print(type(xy))   # <class 'tuple'>
# print(xy.shape) # AttributeError: 'tuple' object has no attribute 'shape'
print(len(xy))    # 2(x, y)

print(xy[0].shape)  # (100, 28, 28, 1)
print(xy[1].shape)  # (100,)

plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7, 7, i + 1)
    plt.imshow(xy[0][i], cmap='gray')
plt.show()