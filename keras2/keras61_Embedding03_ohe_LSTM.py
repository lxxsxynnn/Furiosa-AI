import time
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from tensorflow.keras.utils import to_categorical

# 원핫인코딩 적용하기
# 1. 데이터
docs = [
    '너무 재미있다', '참 최고예요', '참 잘만든 영화예요',
    '추천하고 싶은 영화입니다', '한 번 더 보고 싶어요', '글쎄',
    '별로예요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다', '참 재밌네요',
    'ㅇㅇㅇ 바보', 'ㅁㅁㅁ 잘생겼다', 'ㄴㄴㄴ 또 거짓말한다'
]

labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

# {'참': 1, '너무': 2, '재미있다': 3, '최고예요': 4, '잘만든': 5,
#  '영화예요': 6, '추천하고': 7, '싶은': 8, '영화입니다': 9, '한': 10, 
#  '번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15, 
#  '별로예요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, 
#  '재미없어요': 21, '재미없다': 22, '재밌네요': 23, 'ㅇㅇㅇ': 24, '바보': 25,
#  'ㅁㅁㅁ': 26, '잘생겼다': 27, 'ㄴㄴㄴ': 28, '또': 29, '거짓말한다': 30}

x = token.texts_to_sequences(docs)
print(x)
# [[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], 
#  [10, 11, 12, 13, 14], [15], [16], [17, 18], 
#  [19, 20], [21], [2, 22], [1, 23], 
#  [24, 25], [26, 27], [28, 29, 30]]

from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_x = pad_sequences(x, # 패딩할 대상
                         padding='pre',
                         maxlen = 5,        # 최대 문장 길이
                         )


ohe = OneHotEncoder(sparse_output=False)
encoded_x = ohe.fit_transform(padded_x.reshape(-1, 1))   # (75, 1) > (75, 31)
encoded_x = encoded_x.reshape(-1, 5, 31)                 # (15, 5, 31)

print(encoded_x.shape)   # (15, 5, 31)

y = labels

# 2. 모델 구성
model = Sequential()
model.add(LSTM(10, input_shape=(5, 31)))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'],
              )

start_time = time.time()
model.fit(encoded_x, y,
          epochs=2000,
          batch_size=5,
          )
end_time = time.time()

# 4.평가, 예측
result = model.evaluate(encoded_x, y)
print('loss : ', result[0]) # loss :  4.6343217263711267e-07
print('acc : ', result[1])  # acc :  1.0

y_predict = model.predict(encoded_x)
y_predict = np.round(y_predict)
acc_score = accuracy_score(y, y_predict)
print('acc_score : ', acc_score)        # acc_score :  1.0
print("걸린 시간: ", round(end_time - start_time, 2), " 초")    # 걸린 시간:  25.48 초

text = 'ㅁㅁㅁ 잘생겼다'
x_predict = token.texts_to_sequences([text])

x_predict = pad_sequences(
    x_predict,
    maxlen=5,
    padding='pre'
)

x_predict = ohe.transform(x_predict.reshape(-1, 1))
x_predict = x_predict.reshape(-1, 5, 31)

y_predict2 = model.predict(x_predict)

answer =  accuracy_score([1], np.round(y_predict2))
print('answer : ' , answer) # answer :  1.0