from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# feature_importances_ 분류 : 트리 계열 모델 4개로 iris를 학습하고 컬럼별 중요도 비교
# feature_importances_ : 트리 계열 모델이 fit한 뒤에 생기는 속성 (파라미터 아님, 끝의 _는 학습 후 생기는 값이라는 표시)
# 1. 데이터
x, y = load_iris(return_X_y=True)
print(x.shape, y.shape) # (150, 4) (150,)

random_state = 123

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=random_state,
    stratify=y,
)

# 2. 모델 구성
model1 = DecisionTreeClassifier(random_state=random_state)
model2 = RandomForestClassifier(random_state=random_state)
model3 = GradientBoostingClassifier(random_state=random_state)
model4 = XGBClassifier(random_state=random_state)

models = [model1, model2, model3, model4]

for model in models:
    # 3. 훈련 (sklearn은 compile 없이 fit만)
    model.fit(x_train, y_train)
    print("================", model.__class__.__name__, "================")

    # 4. 평가, 예측
    score = model.score(x_test, y_test)     # 분류에서는 acc, 회귀에서는 r2
    print("acc: ", score)
    print(model.feature_importances_)       # 각 컬럼의 중요도. 합은 1
    # 중요도를 보고 컬럼을 빼거나 새로 만들어 성능을 올리는 게 feature engineering

'''
================ DecisionTreeClassifier ================
acc:  0.8333333333333334
[0.0125     0.03       0.92133357 0.03616643]
================ RandomForestClassifier ================
acc:  0.9333333333333333
[0.08860798 0.0214775  0.48716455 0.40274997]
================ GradientBoostingClassifier ================
acc:  0.9666666666666667
[0.00157033 0.02147603 0.82483537 0.15211826]
================ XGBClassifier ================
acc:  0.9333333333333333
[0.02430454 0.02472077 0.7376847  0.21328993]
'''
# XGBClassifier는 기본값 그대로라 파라미터 튜닝을 해줘야 성능이 올라감
# 컬럼 순서 : 꽃받침 길이, 꽃받침 너비, 꽃잎 길이, 꽃잎 너비
# 네 모델 모두 꽃잎 길이·너비(3·4번째)가 중요도 대부분, 꽃받침 쪽은 0.1 미만