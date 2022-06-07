# Movie Recommendation

## 👋 팀원 소개

|                                                  [김동현](https://github.com/donghyyun)                                                   |                                                                          [임지원](https://github.com/sophi1127)                                                                           |                                                 [이수연](https://github.com/coding-groot)                                                  |                                                                        [진상우](https://github.com/Jin-s-work)                                                                         |                                                                         [심재정](https://github.com/Jaejeong98)                                                                         |
| :-------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![그림1](https://user-images.githubusercontent.com/61958748/172278471-584ffaf5-ea6d-4e63-ae77-7cac4dbae899.png)| ![그림2](https://user-images.githubusercontent.com/61958748/172278474-f2d54e27-898b-4142-af78-b0e370e43ffc.png)| <img width="140" alt="그림3" src="https://user-images.githubusercontent.com/61958748/172278478-f3bbd8ce-3616-4c37-8fa6-4247e20b469e.png">| ![그림4](https://user-images.githubusercontent.com/61958748/172278482-a591c2e4-f4b7-4edf-a390-9e875c2c4226.png)| ![그림5](https://user-images.githubusercontent.com/61958748/172278489-00773bd6-080f-41ec-b828-24f4dabc5f98.png)|    
<br/>

## ✨Contribution

- [`김동현`](https://github.com/donghyyun) &nbsp; MF model • RecVAE

- [`임지원`](https://github.com/sophi1127) &nbsp; lightFM • DeepCTR • DeepFM

- [`이수연`](https://github.com/coding-groot) &nbsp; DeepFM • Ensemble(hard voting)

- [`진상우`](https://github.com/Jin-s-work) &nbsp; SASRec • SSE-PT   

- [`심재정`](https://github.com/Jaejeong98) &nbsp; SAR • Bert4Rec • Ensemble(hard voting)

<br/>

## About Project
![image](https://user-images.githubusercontent.com/61958748/172305015-db2c7c4d-b457-412e-8c39-0e46f7037ef8.png)   
<br/>

### 사용자의 영화 시청 이력 데이터를 바탕으로 사용자가 다음에 시청할 영화 및 좋아할 영화를 예측할 수 있는가? 
- `Timestamp`를 고려한 사용자의 순차적인 이력을 고려하고 `Implicit feedback`을 고려.      
- `Implicit feedback` 기반의 `Sequential recommendation` 시나리오를 바탕으로 사용자의 `Time-ordered sequence`에서 일부 `item`이 누락된 (dropout) 된 상황을 상정.      
- 여러 가지 아이템(영화)과 관련된 `content(side-information)`를 효과적으로 활용.     
<br/>

### 평가 지표
<img width="824" alt="스크린샷 2022-06-07 오후 2 53 09" src="https://user-images.githubusercontent.com/61958748/172305842-0a241881-bac8-428d-b415-8b6f68639a6d.png">

- submisison 파일(csv 형태)에 `Training Data`에 존재하는 전체 유저들에 대해서 각각 10개의 아이템을 추천.     
- 사전에 MovieLens 데이터에서 추출해놓은 `ground-truth` 아이템들을 고려하여 위의 수식과 같이 `Recall@10`을 계산.     
<br/>

## 데이터

| 파일      |   내용     |
| -------- | --------- |
| train_ratings.csv | 주 학습 데이터, `userid`, `itemid`, `timestamp`(초)로 구성(5,154,471행)|
| Ml_item2attributes.json     |   전처리에 의해 생성된 데이터(`item`과 `genre`의 mapping 데이터)      |
| titles.tsv | 영화 제목(6,807행) |
| years.tsv | 영화 개봉년도(6,799행) |
| directors.tsv | 영화별 감독(5,905행) |
| genres.tsv | 영화 장르 (한 영화에 여러 장르가 포함될 수 있다, 15,934행) |
| writers.tsv | 영화 작가 (11,307행) |   
<br/>


## EDA & Feature Engineering & 학습데이터 소개
1. user11의 예시를 통한 데이터 이해 <br/>
<img width="523" alt="스크린샷 2022-06-07 오후 3 09 26" src="https://user-images.githubusercontent.com/61958748/172308049-6db66451-ad02-41a1-966c-128ee7a15e52.png">    <br/>
- 376개의 `리뷰` 작성. (375개의 서로 다른 영화 리뷰 작성)   
- 리뷰 작성 날짜 5일 : `주어진 데이터가 Sequential 데이터인지 의문?` <br/>
'2009-01-01' (248개), '2009-01-02' (66개), '2009-08-24' (13개), '2009-08-25' (36개), '2011-01-12’ (13개).

2. user별로 남긴 영화의 개수와 리뷰를 남긴 날짜 사이의 관계 확인을 통한 Sequential data 판단 <br/>
<img width="713" alt="스크린샷 2022-06-07 오후 3 15 27" src="https://user-images.githubusercontent.com/61958748/172308904-39248379-e582-40d7-b257-899c1a282801.png">   <br/>
- 1일 평균 최소 1개, 최대 1,795개, 평균 54개의 영화 리뷰를 남김.
- user가 영화를 볼 때마다 리뷰를 남긴 것이 아니라 하루에 몰아서 리뷰를 남겼다는 판단.
- 실제로 `Sequential 모델`을 사용할 때보다 `Static 모델`을 사용할 때 성능이 높게 나옴.
<br/>


## Model
- __RecVAE__ : 
  - 기존 `Multi-VAE`에 몇 가지 테크닉을 더한 `collaborative filtering`기반 모델. <br/>
  - 제목에 `implicit data` 기반 `top-n recommendation` 문제에 적합. <br/>
  - 실제 대회에서도 단일 모델로 가장 높은 성능인 0.1499를 보임. <br/>
- __lightFM__ : 
  - `implicit feedback`과 `explicit feedback`에 모두 사용할 수 있는 알고리즘. <br/>
  - `user`와 `item`의 메타데이터(side information)를 기존의 `MF 알고리즘`에 결합하여 새로운 항목과 새로운 사용자로 일반화할 수 있다는 장점. <br/>
  - 실제로 `side information`을 사용했을 때 성능이 낮아졌기 때문에 메타데이터를 사용하지 않음. <br/>
- __SASRec__ : 
  - `Markov chain(MC)`과 `Recurrent Neural Network(RNN)`의 장점을 반영한 모델. <br/>
  - `self-attentive`를 활용한 `sequential recommender`모델에서 하이퍼파라미터를 조정하여 일부 `sequential`한 특징을 가진 데이터를 모델링 하는 데에 사용. <br/>
<br/>

### Ensemble
- 모델 각각이 가지고 있는 장점을 살려서 좋은 결과를 내기 위해 `hard voting` 방식을 통한 `ensemble`을 시도.
- 방식 : `recall@10` 결과에 3-4가지 모델을 활용하여 많이 겹치는 것을 우선적으로 추출한 뒤, 성능이 좋은 모델 순으로 가중치를 부여하여 나머지를 채움.
- 가장 성능이 잘 나온 `ensemble`은 `lightFM`에서 recall@30, `recVAE`에서 recall@30, `SASRec`에서 recall@10을 뽑아 `recVAE`>`lightFM`>`SASRec` 순으로 가중치를 부여한 방식.
<br/>

## Result

<img width="766" alt="스크린샷 2022-06-07 오후 3 29 22" src="https://user-images.githubusercontent.com/61958748/172311114-06ca8712-1438-4ecc-bc0c-53b2b407c5bc.png">

- [`lightFM` 30개, `RecVAE` 30개, `SASRec` 10개]를 뽑아 많이 겹치는 것을 우선적으로 추출하여 `RecVAE`, `lightFM`, `SASRec` 순으로 가중치를 부여하여 `Hard voting` : 0.1536 → 0.1533. <br/>
- [`lightFM`, `RecVAE`, `SASRec`, `MultiVAE` 10개씩] 으로 구성된 4개의 모델을 `RecVAE`, `lightFM`, `MultiVAE`, `SASRec` 순으로 가중치를 부여하여 `Hard voting` : 0.1515 → 0.1532. <br/>
- [`lightFM`, `MultiVAE`, `RecVAE`, `SASRec` 30개씩]을 뽑아 `Soft-voting` : 0.1526 → 0.1528. <br/>



