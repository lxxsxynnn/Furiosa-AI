'''
개 고양이 가중치를 가져와서 모델 완성
데이터는 개 고양이 npy 데이터 사용
'''

import numpy as np
from tensorflow.keras.models import load_model

path = 'C:/study/_save/numpy/kaggle_cat_dog_npy/'

# 1. 데이터
x = np.load(path + 'keras48_cat4.npy') / 255.     # 학습 때 rescale=1./255 했으므로 똑같이 맞춤
print(x.shape)                                  # (1, 100, 100, 3)

# 2. 모델 구성
model = load_model(path + 'keras45_04_catdog.keras')

# 3. 컴파일, 훈련
# 저장된 모델에 구조와 가중치가 다 들어있으므로 다시 훈련하지 않음

# 4. 평가, 예측
predict = model.predict(x)
prob = predict[0][0]

# 학습 때 폴더가 cats, dogs 순서였으므로 0이 고양이, 1이 강아지
print('result : ', predict)
print(f'predict : {prob*100:.2f}% 강아지' if prob > 0.5 else f'predict : {(1-prob)*100:.2f}% 고양이')