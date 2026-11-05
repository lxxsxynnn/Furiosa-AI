from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.utils import all_estimators
import sklearn as sk
import warnings
import numpy as np
import time
warnings.filterwarnings('ignore')

# 여러 개의 데이터로 모델 한 번에 돌리기(회귀)
# 1. 데이터
california = fetch_california_housing(return_X_y=True)
diabetes = load_diabetes(return_X_y=True)

datasets = [california, diabetes]
data_name = ['california', 'diabetes']

# 2. 모델 구성
all = all_estimators(type_filter='regressor')

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333,)

start = time.time()
for index, value in enumerate(datasets):
    x, y = value

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=True, random_state=333, train_size=0.8, 
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print("================", data_name[index],"================")
    model_name = []
    model_acc = []
    
    for name, model in all:
        # print(name)
        try:
            model = model()

            # 3. 훈련
            scores = cross_val_score(model, x_train, y_train, cv=kfold)

            # 4. 평가
            y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
            r2 = np.round(r2_score(y_test, y_predict), 4)

            model_name.append(name)
            model_acc.append(r2)

        except:
            # print(name, '모델 오류')
            continue    # 모델 오류가 있으면 건너뛰기

    model_name = np.array(model_name)
    model_acc = np.array(model_acc)

    max_index = np.argmax(model_acc)
    print('The Best Model: ', model_name[max_index], ', R2: ', model_acc[max_index])

end = time.time()

print('elapsed time: ', round(end - start, 2), 'sec')

'''
================ california ================
The Best Model:  HistGradientBoostingRegressor , R2:  0.7947
================ diabetes ================
The Best Model:  GammaRegressor , R2:  0.4407
elapsed time:  4435.58 sec
'''