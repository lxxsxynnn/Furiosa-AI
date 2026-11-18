import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from keras.applications import VGG16

# 전체 가중치와 훈련되는 가중치 개수 세어 보기
vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3),
)

vgg16.trainable = False

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

print(len(model.weights))   # 30 = 13(vgg) * 2 (w, b) + 4
print(len(model.trainable_weights)) # 4