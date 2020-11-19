# Kaggle_Beginner 
이미 종료된 캐글 대회나 초급자 캐글 대회 참여

#### 머신러닝 파이프라인
1. 데이터 이해(EDA)
2. 평가 척도 이해(경진대회의 문제의도 이해와 비슷)
3. 교차 검증 기법 선정
  - 제공된 데이터 분리 
  - 훈련데이터에 머신러닝 학습, 검증데이터에서 평가척도 점수
  - 위의 내용 반복. 검증데이터에 대한 평균 점수 구하기
4. 피처 엔지니어링
5. 모델 튜닝
  - 머신러닝 모델의 최적 파라미터
6. 앙상블
  - 다수의 모델 사용
  

#### kaggle api 설치
```python
pip install kaggle
kaggle competitions list
```
#### kaggle api에 api키 등록
#### 필요한 데이터 다운로드
```python
kaggle competitions download -c [competition name]
pip install -r kaggle_[competition_name]/requirements.txt
```
  
## 산탄데르 제품 추천 경진대회
https://www.kaggle.com/c/santander-product-recommendation

**고객이 신규로 구매할 제품이 무엇인지** 예측
- 이미 보유하고 있는 제품은 신규 구매로 취급하지 않음
- 지난 달에 보유하고 있는 제품을 이번 달에 해지하는 것 또한 신규 구매로 취급하지 않음 

### 평가 척도 - mapk.py
MAP@7(Mean Average Precision @ 7) :
- 모든 예측 결과물의 Average Precision의 평균 값
- @7은 최대 7개의 금융제품 예측
- 예측의 순서에 매우 예민한 평가 척도 (앞쪽에 예측하는 것이 더 좋은 점수를 받음)

### EDA : 데이터 분석

<span style="color:red"> <b>데이터 분석 노트에 꼼꼼히 작성하는 것이 중요하다</b></span><br>
예시. 
|변수명|내용|데이터타입|특징|변수 아이디어| 
|:------:|:-------:|:-------:|:-------:|:-----:|
|fecha_dato|월별 날짜 데이터|'object'| 2015년 1~6월 데이터가 적음|년도, 월 데이터 별도로 나누기
|ncodpers|고객고유번호|'int64'||학습에는 사용되지 않음
|age|나이|'object'>'int64'|분포 그래프가 중간에 끊김. 아래 숫자가 0~100으로 정렬 안됨|나이 데이터 정수형 정제 필요
|canal_entrada|고객 유입 채널|'object'|알파벳 세글자로 암호화된 유입 경로 변수임. 상위 5개가 대부분을 차지|

### 데이터 전처리
Tabular 형태의 시계열(Time-Series)데이터 제공

### 피처 엔지니어링
Tabular 형태의 시계열 데이터일 경우 딥러닝 모델 보다 트리 기반의 앙상블 모델이 더 좋은 성능 - 시계열 데이터는 과거 데이터를 활용하는 lag 데이터를 파생변수로 생성 <br>
scikit-learn 라이브러리에서 DecisionTree, RandomForest, ExtraTrees, AdaBoost, GradientBoosting 모델 지원 <br>
가장 많이 사용하는 모델 **XGBoost & LightGBM**
