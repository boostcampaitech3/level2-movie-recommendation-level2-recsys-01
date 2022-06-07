# Movie Recommendation

## 👋 팀원 소개

|                                                  [김동현](https://github.com/donghyyun)                                                   |                                                                          [임지원](https://github.com/sophi1127)                                                                           |                                                 [이수연](https://github.com/coding-groot)                                                  |                                                                        [진상우](https://github.com/Jin-s-work)                                                                         |                                                                         [심재정](https://github.com/Jaejeong98)                                                                         |
| :-------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![그림1](https://user-images.githubusercontent.com/61958748/172278471-584ffaf5-ea6d-4e63-ae77-7cac4dbae899.png)| ![그림2](https://user-images.githubusercontent.com/61958748/172278474-f2d54e27-898b-4142-af78-b0e370e43ffc.png)| <img width="140" alt="그림3" src="https://user-images.githubusercontent.com/61958748/172278478-f3bbd8ce-3616-4c37-8fa6-4247e20b469e.png">| ![그림4](https://user-images.githubusercontent.com/61958748/172278482-a591c2e4-f4b7-4edf-a390-9e875c2c4226.png)| ![그림5](https://user-images.githubusercontent.com/61958748/172278489-00773bd6-080f-41ec-b828-24f4dabc5f98.png)|    
<br/>

## ✨Contribution

- [`김동현`](https://github.com/donghyyun) &nbsp; MF model • RecVAE

- [`심재정`](https://github.com/Jaejeong98) &nbsp; SAR • Bert4Rec • Ensemble(hard voting)

- [`이수연`](https://github.com/coding-groot) &nbsp; DeepFM • Ensemble(hard voting)

- [`임지원`](https://github.com/sophi1127) &nbsp; lightFM • DeepCTR • DeepFM

- [`진상우`](https://github.com/Jin-s-work) &nbsp; SASRec • SSE-PT   
<br/>

## About Project
![image](https://user-images.githubusercontent.com/61958748/172305015-db2c7c4d-b457-412e-8c39-0e46f7037ef8.png)   
<br/>

### 사용자의 영화 시청 이력 데이터를 바탕으로 사용자가 다음에 시청할 영화 및 좋아할 영화를 예측할 수 있는가? 
- Timestamp를 고려한 사용자의 순차적인 이력을 고려하고 Implicit feedback을 고려해야 한다. 
- Implicit feedback 기반의 Sequential recommendation 시나리오를 바탕으로 사용자의 Time-ordered sequence에서 일부 item이 누락된 (dropout) 된 상황을 상정한다.
- 여러 가지 아이템(영화)과 관련된 content(side-information)를 효과적으로 활용해야 한다.   
<br/>

### 평가 지표
<img width="824" alt="스크린샷 2022-06-07 오후 2 53 09" src="https://user-images.githubusercontent.com/61958748/172305842-0a241881-bac8-428d-b415-8b6f68639a6d.png">

- submisison 파일(csv 형태)에 Training Data에 존재하는 전체 유저들에 대해서 각각 10개의 아이템을 추천합니다.   
- 사전에 MovieLens 데이터에서 추출해놓은 ground-truth 아이템들을 고려하여 위의 수식과 같이 Recall@10을 계산합니다.   
<br/>
### 데이터

| 파일      |   내용     |
| -------- | --------- |
| train_ratings.csv | 주 학습 데이터, userid, itemid, timestamp(초)로 구성(5,154,471행)|
| Ml_item2attributes.json     |   전처리에 의해 생성된 데이터(item과 genre의 mapping 데이터)      |
| titles.tsv | 영화 제목(6,807행) |
| years.tsv | 영화 개봉년도(6,799행) |
| directors.tsv | 영화별 감독(5,905행) |
| genres.tsv | 영화 장르 (한 영화에 여러 장르가 포함될 수 있음, 15,934행) |
| writers.tsv | 영화 작가 (11,307행) |   
<br/>

