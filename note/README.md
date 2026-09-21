# 스터디 노트

Keras 딥러닝 학습 노트. 날짜별 노트북 한 개 + 그날 실습한 스크립트로 구성됩니다.

| 노트 | 날짜 | 다룬 내용 | 실습 코드 |
|---|---|---|---|
| [day01](day01_0831_keras.ipynb) | 08/31 | 환경 구축, AI 수학 기초, 순전파/역전파, Sequential·Dense, epoch/batch_size | `keras/keras01.py` ~ `keras06_batch.py` |
| [day02](day02_0901_keras.ipynb) | 09/01 | 파이썬 import 방식, 스칼라~텐서, shape, reshape, train/test 분리 | `keras/keras07_matrix.py` ~ `keras09_train_test1.py` |
| [day03](day03_0902_keras.ipynb) | 09/02 | 학습과 추론(NPU), train/test 분리, 시각화, 실전 데이터셋, loss 읽는 법 (작성 중) | `keras/keras09_train_test2.py` ~ `keras11_03_boston_tf.py` |
| [day04](day04_0903_keras.ipynb) | 09/03 | 회귀와 분류, MSE/MAE/RMSE, R², evaluate와 predict, pandas·CSV, 이상치·결측치, 대회 데이터 구조 | `keras/keras12_01_RS_RMSE_boston.py` ~ `keras13_ddarung01.py` |
| [day05](day05_0904_keras.ipynb) | 09/04 | 문자열·경로 다루기, 판다스 기초, 제출 파이프라인, 데이터 누수, 활성화 함수와 ReLU | `keras/keras13_ddarung02_submit.py` ~ `keras14_kaggle_bike1.py` |
| [day06](day06_0907_keras.ipynb) | 09/07 | 딕셔너리, verbose, 검증 데이터(validation), 학습 시간 측정, history 시각화, EarlyStopping | `keras/keras15_verbose.py` ~ `keras20_EarlyStopping5_kaggle_bike.py` |
| [day07](day07_0908_keras.ipynb) | 09/08 | 속성·키 접근, local/global minima, 회귀에서 분류로, stratify, sigmoid, binary_crossentropy와 metrics | `keras/keras21_sigmoid_metrics_cancer.py` |
| [day08](day08_0909_keras.ipynb) | 09/09 | 클래스 인스턴스화·메서드 체이닝, 다중분류, 원핫 인코딩, softmax, argmax | `keras/keras23_softmax1_OneHot_iris.py` ~ `keras23_softmax4_digits.py` |
| [day09](day09_0910_keras.ipynb) | 09/10 | 판다스→넘파이, 이진을 다중으로 풀기, `summary()`와 파라미터 개수, `input_shape`, 스케일링과 그 순서 | `keras/keras24_kaggle_santander.py` ~ `keras28_Scaler10_digits.py` |
| [day10](day10_0911_keras.ipynb) | 09/11 | 스케일링의 목적, 스케일러 4종(MinMax·Standard·MaxAbs·Robust)과 이상치, 모델 저장·불러오기 | `keras/keras29_1_save_model.py` ~ `keras29_4_load_model2.py` |
| [day11](day11_0914_keras.ipynb) | 09/14 | 파일명 문자열 만들기, 가중치 저장·불러오기, ModelCheckpoint, 체크포인트 이력 남기기, Dropout, 함수형 모델 | `keras/keras29_5_save_weights.py` ~ `keras34_function00.py` |
| [day12](day12_0915_keras.ipynb) | 09/15 | GPU 환경 구축(TF 2.9 버전 조합), CPU·GPU 실행 시간 비교, 모델 종류와 데이터 차원, CNN(이미지 수치화, `Conv2D`, MNIST, 이미지 스케일링) | `keras/keras35_gpu_test00.py` ~ `keras36_cnn3_mnist.py` |
| [day13](day13_0916_keras.ipynb) | 09/16 | `Flatten`, `Conv2D`의 filters 인자, padding, strides, MaxPooling | `keras/keras36_cnn1.py` ~ `keras39_MaxPooling_0.py` |
| [day14](day14_0917_keras.ipynb) | 09/17 | Conv2D 파라미터 개수 계산, Global Average Pooling, DNN으로 이미지 처리, 2차원 데이터를 CNN으로 처리 | `keras/keras39_MaxPooling0.py` ~ `keras42_cnn10_digits.py` |
| [day15](day15_0918_keras.ipynb) | 09/18 | ImageDataGenerator와 증폭, fill_mode, 폴더 구조로 x·y 만들기, batch_size와 Iterator, 제너레이터를 모델에 먹이는 법 | `keras/keras44_ImageDataGenerator1.py` ~ `keras44_ImageDataGenerator3_CatDog.py` |

## 주제별 찾아보기

**기초 개념**
- 학습(training) vs 추론(inference), NPU — day03 §1
- 학습이 "반복"인 이유, 경사 하강법 — day01 §3
- 순전파(forward) / 역전파(backward) — day01 §3-3
- AI > ML > DL > LLM 포함 관계 — day01 §4
- 하이퍼파라미터란 — day01 §8, §9
- 회귀와 분류의 차이 — day04 §1
- 로스 = 에러 = 오차 = 코스트 — day04 §1
- 활성화 함수란, 기본값이 `linear`라는 것 — day05 §3
- ReLU가 비선형성을 주는 이유, 기울기 소실 방지 — day05 §3
- local minima와 global minima — day07 §1
- loss 0이 목표가 아닌 이유 — day07 §1
- 이진분류 vs 다중분류 — day07 §2
- sigmoid 함수와 확률로 읽기 — day07 §4
- 다중분류 (라벨 3개 이상) — day08 §1
- 원핫 인코딩이 필요한 이유, 라벨 간 거리 — day08 §2
- 원핫과 임베딩의 차이 — day08 §2 cf)
- softmax의 합이 항상 1인 이유 — day08 §4
- 이진분류를 다중분류로 풀 수 있다 — day09 §1
- 파라미터(w·b) 개수 세는 공식 — day09 §2
- 스케일링이 필요한 이유, 큰 값이 loss를 지배함 — day09 §4
- MinMaxScaler 공식 `(x-MIN)/(MAX-MIN)` — day09 §4
- 부동소수점 오차 `1.0000000000000002` — day09 §4 cf)
- 스케일링의 목적 — 크기만으로 우위를 갖지 못하게 — day10 §1
- GPU 환경 구축 — 드라이버·CUDA·cuDNN·numpy 버전 조합, `nvidia-smi`·`nvcc -V` — day12 §1
- CPU·GPU 실행 시간 비교 — 작은 Dense 모델은 CPU가 빠른 경우가 많음 — day12 §2
- DNN·RNN·CNN과 입력 데이터 차원 (2·3·4차원) — day12 §3
- 이미지 수치화 `(장수, 세로, 가로, 채널)` — day12 §4

**Keras API**
- 코딩 4단계 (데이터 → 모델 → compile/fit → evaluate/predict) — day01 §5
- Sequential / Layer / Node(Unit) — day01 §6
- Dense(완전연결층) — day01 §7
- epoch vs batch_size vs iteration — day01 §10
- `verbose` — 출력 옵션이고 결과에 영향 없음 — day06 §1
- `fit`이 반환하는 History 객체와 `hist.history` — day06 §6
- `evaluate` 안에 `predict`가 포함된다 — day04 §4
- MSE를 구하는 두 경로 (케라스 / sklearn) — day04 §4
- `activation='relu'` 넣는 위치 — day05 §3
- 콜백(callback)과 `EarlyStopping` — day06 §7
- `restore_best_weights`가 필요한 이유 — day06 §7
- 출력층 `activation='sigmoid'` (이진분류) — day07 §4
- `loss='binary_crossentropy'` — day07 §5
- `metrics`와 loss의 차이, 정확도를 loss로 못 쓰는 이유 — day07 §5
- `metrics`를 주면 `evaluate`가 리스트를 반환 — day07 §5
- 출력층 `activation='softmax'` (다중분류) — day08 §4
- `argmax`로 확률을 라벨로 되돌리기 — day08 §5
- `evaluate`에 y를 안 주면 나는 에러 — day08 §5 cf)
- `model.summary()` 읽는 법, `Output Shape`의 `None` — day09 §2, §3
- `input_dim` vs `input_shape`, data shape → input shape — day09 §3
- `scaler.fit()` / `scaler.transform()`의 역할 구분 — day09 §4
- `model.save()` / `load_model()` — `.keras` 포맷 — day10 §5
- 저장 시점(`fit` 전/후)에 따라 달라지는 것 — day10 §5
- 불러온 모델에 `compile`이 필요한 경우 — day10 §5
- `save_weights` / `load_weights` — `.weights.h5`, 구조·compile 없음 — day11 §1
- `ModelCheckpoint` — 훈련 중 최고 지점을 파일로 저장 — day11 §2
- `restore_best_weights`(메모리) vs 체크포인트(파일) — day11 §2
- 체크포인트 파일명에 `{epoch}`·`{val_loss}` 넣기 — day11 §0
- `Dropout` — 배치마다 랜덤하게 노드를 꺼서 과적합 줄이기 — day11 §3
- 함수형 모델 `Input` / `Model` — `Sequential`과 연결 방식 비교 — day11 §4
- `Conv2D` — `kernel_size`(자르는 크기)와 `filters`(필터 개수 = 출력 채널 수) — day12 §4
- `Conv2D` 층별 Output Shape·Param # 계산 — day12 §4
- `Flatten` — Conv 출력을 한 줄로 펴서 `Dense`에 연결, 파라미터 0 — day13 §1
- `padding` — `valid`/`same`, 가장자리 정보와 크기 유지 — day13 §3
- `strides` — 필터 이동 칸수, 출력 크기 축소와 정보 손실 — day13 §4
- `MaxPooling2D` — 구역별 최댓값만 남겨 크기를 절반으로, 파라미터 0 — day13 §5
- Conv2D 파라미터 개수 `f × (n × n × c + b)`, c는 앞 층의 filters — day14 §1
- `GlobalAveragePooling2D` — 특징맵마다 평균 하나, Flatten 대신 써서 Dense 파라미터 감소 — day14 §2
- `AveragePooling2D`와 `GlobalAveragePooling2D`의 차이 — day14 §2 cf)
- DNN으로 이미지 처리 — `reshape(-1, 28 * 28)`, 컬러는 `32 * 32 * 3` — day14 §3
- 표 데이터를 reshape해서 `Conv2D`에 넣기 — 가능한 조건과 성능을 장담 못하는 이유 — day14 §4
- 표 데이터를 CNN으로 바꿀 때 자주 틀리는 곳 (reshape 곱, input_shape, 작은 입력, 출력층 활성화) — day14 §4 주의)
- `ImageDataGenerator` — 이미지를 수치화하고 변형해서 늘리는(증폭) 도구 — day15 §1
- 증폭 옵션 8가지 (`rescale`, flip, shift, `rotation_range`, `zoom_range`, `shear_range`, `fill_mode`) — day15 §1-1
- `fill_mode` 4종 — 변형으로 생긴 빈 공간을 채우는 방법 — day15 §1-2
- 제너레이터를 모델에 먹이는 두 가지 방법 (통째로 꺼내기 / 그대로 넘기기) — day15 §1-6
- 이름이 같은 `batch_size` 두 개 — 꺼내는 단위 vs 갱신 단위 — day15 §1-7 주의)
- 제너레이터에서 검증 데이터 나누기 — 비율 지정이 통하지 않는 이유 — day15 §1-8

**데이터 다루기**
- 실전 데이터셋 불러오기 (`load_` vs `fetch_`, Bunch 객체) — day03 §4
- sklearn과 케라스의 반환 순서 차이 — day03 §4
- 특성 스케일이 제각각이면 생기는 문제 — day03 §5
- CSV를 뷰어에서 고치면 안 되는 이유 — day04 §5
- `pd.read_csv`와 `index_col=0` — day04 §5
- 이상치(outlier)의 기준은 상대적 — day04 §6
- MinMaxScaler는 이상치에 가장 약하다 — day10 §1
- 이상치 처리가 먼저, 스케일링이 나중 — day10 §1
- 스케일링은 범위를 맞추는 것이지 분포를 맞추는 게 아님 — day10 §1
- StandardScaler — 평균 0, 표준편차 1 — day10 §2
- MinMaxScaler와 StandardScaler의 보완 관계 — day10 §2
- 이상치가 있으면 StandardScaler 쪽 — day10 §2
- StandardScaler는 정규분포로 만들어주지 않는다 — day10 §2 cf)
- MaxAbsScaler — 최대 절댓값으로 나누기, 0과 부호 유지 — day10 §3
- 0에서 먼 컬럼은 MaxAbs가 범위를 거의 못 씀 — day10 §3
- RobustScaler — 중앙값과 IQR로 나누기, 이상치에 강함 — day10 §4
- Robust도 이상치를 없애지는 않는다 — day10 §4
- 결과가 좋다 ≠ 스케일러가 좋다 (알게 되는 건 내 데이터) — day10 §4 주의)
- 결측치 처리 3가지 (`dropna` / `fillna` / 앞뒤 채우기) — day04 §6
- `dropna()`가 행 단위로 지운다는 것 — day04 §6 cf)
- train은 `dropna()`, test는 `fillna()`인 이유 — day04 §6
- x / y 분리 (`drop(axis=1)`, `df['count']`) — day04 §7
- `df.isna()` / `df.isnull()`은 같은 함수 — day04 §6
- `to_csv()`로 제출 파일 저장 — day05 §1
- 대회 제출 파이프라인 5단계 — day05 §1
- 예측값을 답안지 컬럼에 대입하기 — day05 §1
- test에 없는 컬럼은 x에 둘 수 없다 — day05 §2
- 데이터 누수(leakage) — `casual` + `registered` = `count` — day05 §2 cf)
- 분류 데이터를 받으면 먼저 라벨 확인 — day07 §3
- 라벨별 개수 세기 (`np.unique` / `value_counts`) — day07 §3
- 클래스 불균형을 확인해야 하는 이유 — day07 §3
- `stratify=y`로 라벨 비율 유지하며 분할 — day07 §3
- 원핫 만드는 3가지 방법과 각각의 주의점 — day08 §3
- `to_categorical`은 라벨이 0부터여야 함 — day08 §3
- `LabelEncoder`는 원핫이 아니라 그 전 단계 — day08 §3 cf)
- `stratify`에 원핫(2차원)을 넘겨도 되는가 — day08 §6 cf)
- 판다스를 넘파이로 바꾸는 3가지 — day09 §0
- softmax 결과에서 제출할 확률 열 고르기 `[:, 1]` — day09 §1
- 스칼라 / 벡터 / 행렬 / 텐서 — day02 §1
- shape 읽는 법 — day02 §2
- reshape와 `-1` — day02 §3
- reshape vs 전치(`.T`) 차이 — day02 §3 cf)
- `input_dim`은 열(특성) 개수 — day02 §3
- `mnist.load_data()` — train/test로 나눠 반환, `(60000, 28, 28)`, `plt.imshow`로 확인 — day12 §4
- 이미지 스케일링 `/255.`(0~1)과 `(x-127.5)/127.5`(-1~1) — day12 §4
- `pd.value_counts`는 pandas 3.0에서 삭제 → `pd.Series(y).value_counts()` — day12 §4
- 폴더 구조로 x와 y 만들기 — 하위 폴더 이름이 라벨이 됨 — day15 §1-4
- 훈련 데이터만 증폭하고 테스트는 스케일링만 하는 이유 — day15 §1-3
- `batch_size`와 Iterator — 인덱스는 이미지 번호가 아니라 배치 번호 — day15 §1-5
- 평가용 데이터는 섞지 않는다 — 예측 순서와 정답 순서 — day15 §1-9 주의)

**학습 방법론**
- 과적합(overfitting)이 생기는 이유 — day02 §4
- train / test 분리, shuffle이 필요한 이유 — day02 §4
- 나누는 방법 3가지 (직접/슬라이싱/`train_test_split`) — day03 §2
- `random_state`를 고정하는 이유 — day03 §2
- 시계열은 섞으면 안 되는 이유 — day03 §2
- `fit`의 loss와 `evaluate`의 loss 차이 — day03 §5
- loss 값을 데이터셋끼리 비교할 수 없는 이유 — day03 §5
- matplotlib으로 회귀선 그려 확인하기 — day03 §3
- 검증 데이터(validation)가 필요한 이유 — day06 §2
- val과 test를 나누는 이유 — day06 §2
- 검증셋 만드는 4가지 방법 — day06 §3
- `validation_split` vs `validation_data` — day06 §3
- loss/val_loss 그래프로 과적합 읽기 — day06 §6
- 학습 시간 재기 (`time.time()`) — day06 §5
- 층·epoch만 키운다고 좋아지지 않는다 (튜닝 기록) — day05 §4
- 과적합 지점에서 자동으로 멈추기 — day06 §7
- `patience`는 epochs의 10~20% 선 — day06 §7
- Early Stopping이 최고 모델을 보장하지 않는 이유 — day07 §1
- 분류에 R²·RMSE를 쓰면 안 되는 이유 — day07 §5 cf)
- validation / x_test / test.csv 3단계 — day04 §7
- `x_test`(내 채점용)와 `test_csv`(제출용)의 차이 — day04 §7
- 스케일링은 분리 뒤에, `fit`은 train에만 — day09 §5
- 스케일러를 전체에 `fit`하면 생기는 누수 — day09 §5
- x_test가 0~1을 벗어나는 것은 정상 — day09 §5
- `validation_split`에 남아 있는 미세한 누수 — day09 §5 cf)

**평가 지표**
- MSE가 "제곱"인 이유, 경사 하강법 갱신식 — day03 cf)
- MAE 대신 MSE를 쓰는 이유 (기울기·미분 가능성) — day04 §2
- RMSE는 단위만 되돌린다 (이상치 취약성은 남음) — day04 §2
- 케라스에 RMSE가 없어 직접 함수로 만드는 것 — day04 §0, §2
- R²(결정계수) 읽는 법, 0이 기준선인 이유 — day04 §3
- R²가 지나치게 높을 때 의심할 것 — day04 §3
- 불균형 데이터에서 accuracy가 쓸모없는 이유 — day09 §1

**파이썬 / 문법**
- `from A import B`를 쓰는 이유 — day02 §0
- 함수 vs 클래스, PEP8 이름 규칙 — day02 §0
- REPL vs 스크립트 실행 — day01 §0
- 슬라이싱 `[시작:끝]`, 끝 인덱스는 미포함 — day03 §0
- 튜플 언패킹 `(a, b), (c, d) = ...` — day03 §0
- 문자열 연결로 경로 만들기, 이스케이프 `\\` — day05 §0
- raw string은 백슬래시로 끝날 수 없다 — day05 §0 cf)
- `datetime.strftime()` 포맷 코드 — day05 §0
- 여러 줄 문자열 `"""..."""` — day05 §0
- 딕셔너리(dict) — 키로 값 꺼내기, `.keys()` / `.get()` — day06 §0
- 속성 접근 `.` vs 키 접근 `[]`, `Bunch`가 둘 다 되는 이유 — day07 §0
- `type()`으로 자료형 확인하기 — day07 §0 cf)
- 클래스 인스턴스화 `Class()` vs 클래스 자체 — day08 §0
- 메서드 체이닝 `a.b().c()` — day08 §0
- 속성 `.values` vs 메서드 `.to_numpy()` — day09 §0
- `def` / `return` 기본 문법 — day04 §0
- 윈도우 경로에서 `\\`와 `/` — day04 §5
- `datetime.datetime` — 모듈과 클래스 이름이 같음 — day11 §0
- `''.join([...])`로 문자열 이어붙이기 — day11 §0
- 포맷 지정자 `{:04d}` / `{:.4f}`, `4f`와 `.4f`의 차이 — day11 §0

## 작성 규칙

- 노트북 제목은 `#`, 대단원은 `##`, 소단원은 `###`
- 요약이 아니라 **개념·설명 위주**로 씁니다. 섹션 안의 순서는 개념 한 줄 → **필요한 이유** → **종류·규칙**(번호) → 설명·예제
  - 종류·규칙은 `1. 이름 : 설명` 형태로 번호를 매기고, 세부 설명은 `-` 하위 목록으로
  - 실습 결과표에는 **결과 해석**(왜 그런 결과인지)을 붙입니다
- 섹션 번호: 대단원 `## N. 개념명`, 소단원 `### N-M. 개념명` (예: `### 3-3. 순전파(Forward)와 역전파(Backward)`)
  - 곁가지는 `cf)`, 함정은 `주의)`, 예외는 `예외)`를 개념명 앞에 붙입니다
  - README의 `§N` 참조가 깨지지 않도록 대단원 번호는 바꾸면 README도 같이 고칩니다
- 실행 결과는 저장하지 않고, 기대값을 코드 주석으로 적습니다
- 그림은 `assets/`에 두고 이미지로 참조. 직접 그린 다이어그램은 SVG, 캡처·사진은 PNG (노트북에 인라인 HTML/SVG를 넣지 않음)
- 줄바꿈은 `<br/>` 대신 빈 줄(문단 구분)로
