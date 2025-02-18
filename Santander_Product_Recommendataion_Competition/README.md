# BaseLine

## 데이터 불러오기

```python
trn = pd.read_csv('./input/train_ver2.csv')
tst = pd.read_csv('./input/test_ver2.csv')
```

## 데이터 전처리
- 결측값
- 불필요한 값(유령 고객)
- train과 test 데이터 통합(concat으로)

```python
# 제품 변수 결측값을 미리 0으로 대체한다.
trn[prods] = trn[prods].fillna(0.0).astype(np.int8)

# 24개 제품 중 하나도 보유하지 않는 고객 데이터를 제거한다.
no_product = trn[prods].sum(axis=1) == 0
trn = trn[~no_product]

# 훈련 데이터와 테스트 데이터를 통합한다. 테스트 데이터에 없는 제품 변수는 0으로 채운다.
for col in trn.columns[24:]:
    tst[col] = 0
df = pd.concat([trn, tst], axis=0)
```

- 학습에 사용할 변수 list에 담기

|범주형 변수|수치형 변수|
|------|------|
|.factorize()함수를 통한 label encoding|특이값과 결측값을 -99로 변환, 정수로 변환|


✔ concat과 merge의 차이점  
- merge: 공통된 항목 기준으로 합침
- concat: 단순히 합칠때 사용

✔ .factorize()을 하면 label encoding을 얻을 수 있음. 출력값으로는 encoding 된 값과 class의 값들이 나온다.


## 피처 엔지니어링 (파생변수 생성)

- 년도와 월 정보 추출

```python
# (피쳐 엔지니어링) 두 날짜 변수에서 연도와 월 정보를 추출한다.
df['fecha_alta_month'] = df['fecha_alta'].map(lambda x: 0.0 if x.__class__ is float else float(x.split('-')[1])).astype(np.int8)
df['fecha_alta_year'] = df['fecha_alta'].map(lambda x: 0.0 if x.__class__ is float else float(x.split('-')[0])).astype(np.int16)
features += ['fecha_alta_month', 'fecha_alta_year']
```

- lag-1 데이터 생성

```python
# 날짜를 숫자로 변환하는 함수이다. 2015-01-28은 1, 2016-06-28은 18로 변환된다
def date_to_int(str_date):
    Y, M, D = [int(a) for a in str_date.strip().split("-")] 
    int_date = (int(Y) - 2015) * 12 + int(M)
    return int_date

# 날짜를 숫자로 변환하여 int_date에 저장한다
df['int_date'] = df['fecha_dato'].map(date_to_int).astype(np.int8)

# 데이터를 복사하고, int_date 날짜에 1을 더하여 lag를 생성한다. 변수명에 _prev를 추가한다.
df_lag = df.copy()
df_lag.columns = [col + '_prev' if col not in ['ncodpers', 'int_date'] else col for col in df.columns ]
df_lag['int_date'] += 1

# 원본 데이터와 lag 데이터를 ncodper와 int_date 기준으로 합친다. Lag 데이터의 int_date는 1 밀려 있기 때문에, 저번 달의 제품 정보가 삽입된다.
df_trn = df.merge(df_lag, on=['ncodpers','int_date'], how='left')

# lag-1 변수를 추가한다.
features += [feature + '_prev' for feature in features]
features += [prod + '_prev' for prod in prods]
print(features)
```

## 모델 학습
- **교차검증**을 위한 데이터분리
    - 신규 구매 건수만 추출
    
```python
for i, prod in enumerate(prods):
    prev = prod + '_prev'
    prX = trn[(trn[prod] == 1) & (trn[prev] == 0)]
    prY = np.zeros(prX.shape[0], dtype=np.int8) + i
    X.append(prX)
    Y.append(prY)
XY = pd.concat(X)
Y = np.hstack(Y)
XY['y'] = Y
```

- 모델 : XGBoost  
```XGBoost``` 파라미터 설명

| 파라미터 | 설명 | 비고 |
|:------|:-------:|:-----------:|
|max_depth|트리 모델의 최대 깊이|값이 높을 수록 복잡한 트리 모델, 과적합의 원인|
|eta|learning rate와 같은 개념. 0과 1사이의 값|값이 너무 높으면 학습이 잘 되지 않으며, 너무 낮으면 학습이 느려짐|
|colsample_bytree|트리 생성할 때 훈련 데이터에서 변수를 샘플링해주는 비율.  모든 트리는 전체 변수의 일부만을 학습하여 서로의 약점을 보완| 보통 0.6~0.9 사용|
|colsample_bylevel|트리의 레벨 별로 훈련 데이터의 변수를 샘플링해주는 비율| 보통 0.6~0.9 사용|

**파라미터 튜닝 작업에 많은 시간을 쏟지 말고 피처 엔지니어링에 더 많은 시간을 쏟을 것을 권장한다**  
적당한 수준의 피처 엔지니어링을 통해 얻은 변수와 엄청난 수준의 파라미터 튜닝을 통해 얻은 하나의 완벽한 모델보다, 적당한 수준의 파라미터 튜닝을 진행한 모델과 많은 시간을 피처 엔지니어링에 투자하여 얻어낸 양질의 변수를 학습한 모델이 보편적으로 더 좋은 성능을 보임

- XGBoost 형태  
```python
# 훈련, 검증 데이터를 XGBoost 형태로 변환한다.
X_trn = XY_trn[features].values
Y_trn = XY_trn['y'].values
dtrn = xgb.DMatrix(X_trn, label=Y_trn, feature_names=features)

X_vld = XY_vld[features].values
Y_vld = XY_vld['y'].values
dvld = xgb.DMatrix(X_vld, label=Y_vld, feature_names=features)

# XGBoost 모델을 훈련 데이터로 학습한다
watch_list = [(dtrn, 'train'), (dvld, 'eval')]
model = xgb.train(param, dtrn, num_boost_round=1000, evals=watch_list, early_stopping_rounds=20)
```

- 모델 저장

```python
# 학습한 모델을 저장한다.
import pickle
pickle.dump(model, open("./model/xgb.baseline.pkl", "wb"))
best_ntree_limit = model.best_ntree_limit  #early stopping
```
