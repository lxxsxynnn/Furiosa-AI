import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Flatten, Bidirectional, Dropout, Embedding
from sklearn.metrics import accuracy_score

# 임베딩 실습 - 
# 1. 데이터
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000,     # 단어사전의 개수, 빈도수가 높은 단어 순으로 1000개 뽑기
    # maxlen=1000,          # 50단어 이상인 기사는 제외 (자르는 게 아님). 남은 기사도 길이가 달라 패딩 필요
    # test_split=0.2,   # imdb에서는 이미 train/test 나뉘어있어서 필요 없음
)

print(x_train.shape, y_train.shape) # (25000,) (25000,)
print(x_test.shape, y_test.shape)   # (25000,) (25000,)
print(np.unique(y_train))           # [0 1]

avglen = sum(map(len, x_train))/len(x_train)
avglen2 = sum(map(len, x_test))/len(x_test)

print('train_set average length : ', avglen)    # train_set average length :  238.71364
print('test_set average length : ', avglen2)    # test_set average length :  230.8042

# x 패딩 추가
padded_x_train = pad_sequences(x_train, padding='pre',
                               maxlen=250,
                               truncating='pre',
                               )

padded_x_test = pad_sequences(x_test, padding='pre',
                              maxlen=250,
                              truncating='pre',
                             )

print(padded_x_train.shape) # (25000, 250)
print(padded_x_test.shape)  # (25000, 250)

# 2. 모델 구성
# Embedding + LSTM / GRU
# Embedding + Bidirectional
# Embedding + Flatten + DNN

model = Sequential()
model.add(Embedding(10000, 100))
model.add(Bidirectional(GRU(64)))         # (N, 250, 100) > (N, 128) - Bidirectional을 사용하면 64 + 64 > 128
model.add(Dropout(0.3))
model.add(Dense(500, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'],
              )
model.fit(padded_x_train, y_train,
          epochs=1000,
          batch_size=512,
          )

# acc 0.9 이상
# 4. 평가, 예측
result = model.evaluate(padded_x_test, y_test)
print('loss : ', result[0]) # loss :  5.668089389801025
print('acc : ', result[1])  # acc :  0.849399983882904

y_predict = model.predict(padded_x_test)
y_predict = np.round(y_predict)
print('acc_score : ', accuracy_score(y_test, y_predict))    # acc_score :  0.8494