# **Prefix Tuning**

## **Processed Data Source Link**

**Link: [Google Drive](https://drive.google.com/file/d/1kUecR7xO7bsHFmUI6AExtY5u2XXlObOG/view)**

원문에서 이메일, URL, 여러 형태의 공백 등 불필요 문자를 제거하고 숫자 사이에 공백을 추가하는 등의 전처리를 시행합니다.

>processed_wiki_ko.txt			한국어 위키백과

>processed_ratings_train.txt	네이버 영화 말뭉치 학습셋 (극성 레이블 있음)

>processed_ratings_test.txt		네이버 영화 말뭉치 테스트셋 (극성 레이블 있음)


**DOWNLOAD PATH:**

	/data/ko_processed_data/

---

## **submodule**

- SentEval

$ ```git submodule init```

$ ```git submodule update```


---

## **conda environment**

$ ```conda env create -f environment.yaml```

- Modify {ENV_NAME} and {USER_NAME}

---

## **How to run**

$ ```cd ./sourcecode/SentEval/data/downstream/```

$ ```./get_transfer_data.bash```

- Download SentEval data

$ ```cd ../../../```

- Working directory: prefix_tuning/sourcecode/

$ ```python ./1_Data_preprocessing.py```

- Load data and convert to json string

$ ```CUDA_VISIBLE_DEVICES={Multiple GPU NUMBERS} python -m torch.distributed.launch --nproc_per_node={NUMBER of GPUs} ./2_Pretraining_prefix_prompts.py``` 

- Pretrain prefix prompts (DDP)

- or $ ```CUDA_VISIBLE_DEVICES={Single GPU NUMBER} python ./2_Pretraining_prefix_prompts.py.py```

- if socket binding error: use option ```--master_port {PORT_ID}```

$ ```python ./3_Freeze_parameters_roberta.py```

- Freeze roberta-base model

$ ```python ./4_Freeze_parameters_prefix_model.py```

- Freeze prefix prompting model

$ ```python ./5_Fine_tuning_roberta.py```

- Fine-tuning roberta-base model

$ ```python ./6_Fine_tuning_prefix_model.py```

- Fine-tuning prefix prompting model

---

## **Data Source:**

Preprocessed Korean Corpus Data

* https://github.com/ratsgo/embedding

* https://ratsgo.github.io/embedding/preprocess.html

* https://ratsgo.github.io/embedding/downloaddata.html
