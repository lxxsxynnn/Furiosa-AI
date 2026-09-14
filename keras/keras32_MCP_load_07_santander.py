# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler,RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 데이터
data_path = "C:/study/_data/kaggle_santander/"

train_csv = pd.read_csv(data_path + "train.csv", index_col=0)
test_csv = pd.read_csv(data_path + "test.csv", index_col=0)
submission_csv = pd.read_csv(data_path + "sample_submission.csv", index_col=0)

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']
print(x.shape, y.shape) # (200000, 200) (200000,)

print(np.unique(y, return_counts=True)) # (array([0, 1]), array([179902,  20098]))

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=100, train_size=0.7,
                                                    stratify=y)

# 2. 모델 구성
save_path = 'C:/study/_save/keras31/'
model = load_model(save_path + 'k31_0914_1436_07_santander_0011-0.2430.keras')

# 3. 컴파일, 훈련

# # 4. 평가, 예측
loss = model.evaluate(x_test, y_test)

print('loss : ', loss[0])
print('acc : ', round(loss[1], 4))

y_predict = model.predict(x_test)

y_predict = np.round(y_predict)

acc_score = accuracy_score(y_test, y_predict)

print('acc_score : ', acc_score)

'''
loss :  0.24428561329841614
acc :  0.9104
acc_score :  0.9104333333333333
'''

# y_submit = model.predict(test_csv)
# submission_csv['target'] = np.round(y_submit).astype(int)

# from datetime import datetime
# now = datetime.now().strftime('%m%d_%H%M')
# submission_csv.to_csv(data_path + "submit/" + "submit_" + now + ".csv")