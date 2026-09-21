import numpy as np
from tensorflow.keras.models import load_model

path = 'C:/study/_save/numpy/man-woman_npy/'

# 1. 데이터
x = np.load(path + 'keras47_es.npy') / 255.     # 학습 때 rescale=1./255 했으므로 똑같이 맞춤
print(x.shape)                                  # (1, 100, 100, 3)

# 2. 모델 구성
model = load_model(path + 'keras47_03_gender.keras')

# 3. 컴파일, 훈련
# 저장된 모델에 구조와 가중치가 다 들어있으므로 다시 훈련하지 않음

# 4. 평가, 예측
predict = model.predict(x)
prob = predict[0][0]

print('result : ', predict)
print(f'predict : {prob*100:.2f}% 여성' if prob > 0.5 else f'predict : {(1-prob)*100:.2f}% 남성')