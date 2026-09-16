from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

# Conv2D 파라미터 알아보기
model = Sequential()
model.add(Conv2D(10, (3, 3), input_shape=(10, 10, 1)))       # 여기서 (3, 3)는 kernel의 크기
model.add(Conv2D(5, (2, 2)))

'''
filters - 출력 채널 수
kernel size - 한 번에 볼 영역 크기 (w * h)
input_shape(h, w, c) - 입력 한 장의 모양(세로, 가로, 채널)
'''