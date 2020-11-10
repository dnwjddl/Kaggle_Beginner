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
  
## 산탄데르 제품 추천 경진대회
**고객이 신규로 구매할 제품이 무엇인지** 예측
- 이미 보유하고 있는 제품은 신규 구매로 취급하지 않음
- 지난 달에 보유하고 있는 제품을 이번 달에 해지하는 것 또한 신규 구매로 취급하지 않음

### 평가 척도 - mk
MAP@7(Mean Average Precision @ 7) : mapk.py
- 모든 예측 결과물의 Average Precision의 평균 값
- @7은 최대 7개의 금융제품 예측
- 예측의 순서에 매우 예민한 평가 척도 (앞쪽에 예측하는 것이 더 좋은 점수를 받음)
