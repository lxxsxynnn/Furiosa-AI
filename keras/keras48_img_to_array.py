from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

path = 'C:/study/_data/image/'

img = load_img(path + 'image.png', target_size=(100, 100))    # keras45_03의 SIZE와 같게

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
print(arr.shape)    # (1, 100, 100, 3)

np_path = 'C:/study/_save/numpy/kaggle_cat_dog_npy/'

np.save(np_path + 'keras48_cat4.npy', arr=arr)