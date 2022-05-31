# **Prefix Prompting**

**모델의 일반화 및 경량화가 가능한 한국어 자연언어처리 모델을 구성하기 위해**

**가상토큰인 Prefix를 추가하여 Pre-trained Prefix Prompting 모델을 구축하고 학습시켜 한국어 감성분류 태스크에 적용하는 방법을 보인다.**

![fig2](https://user-images.githubusercontent.com/62884475/167860566-8f8d8c77-c57a-4044-8d1f-be121c1904ad.png)

## **Processed Data Source Link**

**Link: [Google Drive](https://drive.google.com/file/d/1kUecR7xO7bsHFmUI6AExtY5u2XXlObOG/view)**

원문에서 이메일, URL, 여러 형태의 공백 등 불필요 문자를 제거하고 숫자 사이에 공백을 추가하는 등의 전처리를 시행하였으며 다음 데이터셋 활용

>processed_wiki_ko.txt			한국어 위키백과

>processed_ratings_train.txt		네이버 영화 말뭉치 학습셋 (극성 레이블 있음)

>processed_ratings_test.txt		네이버 영화 말뭉치 테스트셋 (극성 레이블 있음)

---

## **Process**

### **1. Download Data**

제시된 한국어 전처리 데이터를 아래 경로에 다운로드

	/data/ko_processed_data/

---

## **2. Initialize Submodule**

SentEval 서브모듈 정의

```console
$ git submodule init

$ git submodule update
```

---

## **3. Set Environment**

가상환경 생성

```console
$ conda env create -f environment.yaml
```
Modify {ENV_NAME} and {USER_NAME}

SentEval 데이터 다운로드

```console
$ cd ./sourcecode/SentEval/data/downstream/

$ ./get_transfer_data.bash
```

작업 디렉토리 이동 (sourcecode/)

```console
$ cd ../../../
```

Convert json string

```console
$ python ./1_Data_preprocessing.py
```

---

## **4. Train Prefix Prompts**

Prefix 사전 학습

```console
$ CUDA_VISIBLE_DEVICES={Multiple GPU IDs} python -m torch.distributed.launch --nproc_per_node={NUMBER of GPUs} ./2_Pretraining_prefix_prompts.py
```
or

```console
$ CUDA_VISIBLE_DEVICES={Single GPU ID} python ./2_Pretraining_prefix_prompts.py
```

---

## **5. Evaluate Models**

RoBERTa 매개변수 고정 후 성능 측정

```console
$ python ./3_Freeze_parameters_roberta.py

$ python ./4_Freeze_parameters_prefix_model.py
```

Fine-tuning 성능 측정

```console
$ python ./5_Fine_tuning_roberta.py

$ python ./6_Fine_tuning_prefix_model.py
```

| 	Model            			| Accuracy (%) 	|
| ----------------------------- | -------------	|
| RoBERTa            		 	| 62.97    		|
| Prefix-Length10	 		 	| 61.85    		|
| Prefix-Length50	 		 	| 68.31    		|
| Prefix-Length100	 		 	| 64.55    		|
| Prefix-Length200	 		 	| 67.97    		|
| | |
| RoBERTa          (Fine-tuning)| 90.78    		|
| Prefix-Length10  (Fine-tuning)| 90.83    		|
| Prefix-Length50  (Fine-tuning)| 90.89    		|
| Prefix-Length100 (Fine-tuning)| 90.88    		|
| Prefix-Length200 (Fine-tuning)| 90.68    		|
---

## **References**

Preprocessed Korean Corpus Data

* https://github.com/ratsgo/embedding

* https://ratsgo.github.io/embedding/preprocess.html

* https://ratsgo.github.io/embedding/downloaddata.html
