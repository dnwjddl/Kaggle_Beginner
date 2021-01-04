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


### 피처 엔지니어링

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


