from tensorflow.keras.preprocessing.text import Tokenizer

# 두 개의 문장을 토큰화해보기
text1 = 'I just ate very very tasty gimbap really really fast.'
text2 = 'Anna loves trains. Tom lies. Jerry really really often lies.'
token = Tokenizer()
token.fit_on_texts([text1, text2])

print(token.word_index)
# {'really': 1, 'very': 2, 'lies': 3, 'i': 4, 'just': 5, 'ate': 6, 'tasty': 7, 'gimbap': 8, 'fast': 9, 'anna': 10, 'loves': 11, 'trains': 12, 'tom': 13, 'jerry': 14, 'often': 15}

x = token.texts_to_sequences([text1, text2])
print(x)
# [[4, 5, 6, 2, 2, 7, 8, 1, 1, 9], [10, 11, 12, 13, 3, 14, 1, 1, 15, 3]]

# 원핫 인코딩
import numpy as np

# 가공 방법 1. 문장 길이가 같을 때 : 2차원으로 만든 뒤 펴기
x = np.array(x) # 문장 길이가 다를 때 사용하면 에러 발생
print(x.shape)  # (2, 10)
x = x.reshape(-1)
print(x.shape)  # (20,)

# 가공 방법 2. 문장 길이가 다를 때 : 문장별로 꺼내 이어 붙이기
# x1 = np.array(x[0])
# x2 = np.array(x[1])
# print(x1.shape) # (10,)
# print(x2.shape) # (10,)
# x = np.concatenate([x1, x2])
# print(x.shape)  # (20,)

# sklearn
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
x = x.reshape(-1, 1)      # OneHotEncoder가 2차원 데이터만 받아서 모양을 맞춰줘야 함
x = ohe.fit_transform(x)
print(x.shape)  # (20, 15)