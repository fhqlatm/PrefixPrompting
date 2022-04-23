import os
import json

PATH_DIR_SOURCE = '../nsmc'
PATH_DIR_OUTPUT = '../data/nsmc_data'
raw_train = 'ratings_train.txt'
raw_test = 'ratings_test.txt'
prepro_train = 'nsmc_train.json'
prepro_test = 'nsmc_test.json'

raw_files = [raw_train, raw_test]
prepro_files = [prepro_train, prepro_test]

for raw_file, prepro_file in zip(raw_files, prepro_files):
	PATH_raw_file = os.path.join(PATH_DIR_SOURCE, raw_file)

	with open(PATH_raw_file, 'r') as f1:
		fdata = f1.readlines()
		fdata = list(map(lambda s: s.strip(), fdata))	# strip(): 개행문자 등 공백 제거
		prepro_data = []

		for sdata in fdata[1:]:
			sdata = sdata.split('\t')
			data = []

			data.append(sdata[0])
			data.append(sdata[1])
			data.append(int(sdata[2]))
			
			prepro_data.append(data)
				
		PATH_prepro_file = os.path.join(PATH_DIR_OUTPUT, prepro_file)

		with open(PATH_prepro_file, 'w') as f2:
			json.dump(prepro_data, f2, ensure_ascii=False, indent='\t')



