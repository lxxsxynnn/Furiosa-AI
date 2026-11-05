from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import sklearn as sk
import warnings
import numpy as np
import time
warnings.filterwarnings('ignore')

# 여러 개의 데이터로 모델 한 번에 돌리기(분류)
# 1. 데이터
iris = load_iris(return_X_y=True)
cancer = load_breast_cancer(return_X_y=True)
wine = load_wine(return_X_y=True)
digits = load_digits(return_X_y=True)

datasets = [iris, cancer, wine, digits]
data_name = ['iris', 'cancer', 'wine', 'digits']

# 2. 모델 구성
all = all_estimators(type_filter='classifier')

n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333,)

start = time.time()
for index, value in enumerate(datasets):
    x, y = value

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=True, random_state=333, train_size=0.8, 
        stratify=y,
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
            # print("acc: ", scores, '\n평균 acc: ', round(np.mean(scores), 2))

            # 4. 평가
            y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
            acc = np.round(accuracy_score(y_test, y_predict), 4)

            model_name.append(name)
            model_acc.append(acc)

        except:
            # print(name, '모델 오류')
            continue    # 모델 오류가 있으면 건너뛰기

    model_name = np.array(model_name)
    model_acc = np.array(model_acc)

    max_index = np.argmax(model_acc)
    print('The Best Model: ', model_name[max_index], ', ACC: ', model_acc[max_index])

end = time.time()

print('elapsed time: ', round(end - start, 2), 'sec')

'''
================ iris ================
The Best Model:  RandomForestClassifier , ACC:  1.0
================ cancer ================
The Best Model:  AdaBoostClassifier , ACC:  0.9825
================ wine ================
The Best Model:  ExtraTreesClassifier , ACC:  1.0
================ digits ================
The Best Model:  SVC , ACC:  0.95
elapsed time:  124.93 sec
'''