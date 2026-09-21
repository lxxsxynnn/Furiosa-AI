from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# 데이터 증폭하기
path = 'C:/study/_data/image/'

img = load_img(path + 'keras48_2.jpg', target_size=(150, 150))

print(img)          # <PIL.Image.Image image mode=RGB size=150x150 at 0x1E8C7D89FF0>
print(type(img))    # <class 'PIL.Image.Image'>

# plt.imshow(img)
# plt.show()

# img가 클래스 안에 들어있어서 바로 사용하지 못함 > 이미지를 수치화
arr = img_to_array(img)
print(arr)
print(type(arr))    # <class 'numpy.ndarray'>

arr = np.expand_dims(arr, axis=0)   # 차원 증가
print(arr)
print(arr.shape)    # (1, 150, 150, 3)

# np_path = 'C:/study/_save/numpy/kaggle_cat_dog_npy/'

# np.save(np_path + 'keras48_cat4.npy', arr=arr)

########## 증폭 ##########
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

it = data_gen.flow(arr,
                   batch_size=1,
                  )

print(it)           # <keras.preprocessing.image.NumpyArrayIterator object at 0x000001B207D6DED0>
# print(it.next())    # 케라스가 따로 둔 메서드. 케라스 3에서 없어짐
print(next(it))     # 파이썬 표준 방식

print(next(it).shape)   # (1, 150, 150, 3)

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(5,5))

for i in range(5):
    # batch = it.next()
    batch = next(it)
    # print(batch.shape)
    batch = batch.reshape(150, 150, 3)

    ax[i].imshow(batch)
    ax[i].axis('off')

plt.show()