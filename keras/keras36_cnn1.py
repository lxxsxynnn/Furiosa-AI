from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

# cnn 모델 기본 구성
model = Sequential()
model.add(Conv2D(10, (3, 3), input_shape=(10, 10, 1)))       # 여기서 (3, 3)는 kernel의 크기
model.add(Conv2D(5, (2, 2)))

model.summary()
'''
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 8, 8, 10)          100       
                            마지막 요소는 아웃풋 개수                
 conv2d_1 (Conv2D)           (None, 7, 7, 5)           205                                          
=================================================================
Total params: 305
Trainable params: 305
Non-trainable params: 0
'''