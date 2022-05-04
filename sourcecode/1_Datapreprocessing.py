import os
import json

PATH_DIR_SOURCE = '../data/ko_processed_data'
PATH_DIR_OUTPUT = '../data/nsmc_data'

PATH_FILE_TRAIN_SOURCE = 'processed_ratings_train.txt'
PATH_FILE_TEST_SOURCE = 'processed_ratings_test.txt'

PATH_FILE_TRAIN_OUTPUT = 'nsmc_train.json'
PATH_FILE_TEST_OUTPUT = 'nsmc_test.json'
PATH_FILE_DEV_OUTPUT = 'nsmc_dev.json'

def txt_to_json(source, output, mode):
	PATH_FILE_TXT = os.path.join(PATH_DIR_SOURCE, source)
	PATH_FILE_JSON = os.path.join(PATH_DIR_OUTPUT, output)
	PATH_FILE_DEV = os.path.join(PATH_DIR_OUTPUT, PATH_FILE_DEV_OUTPUT)

	data = []

	with open(PATH_FILE_TXT, "r") as f:
		fdata = f.readlines()

		for sdata in fdata:
			processed_data = []
			sdata = sdata.split('\u241E')
			processed_data.append(sdata[0])
			processed_data.append(int(sdata[1][0]))
		
			data.append(processed_data)

	if mode == 'train':
		with open(PATH_FILE_DEV, 'w') as f_dev:
			json.dump(data[:15000], f_dev, ensure_ascii=False, indent='\t')
		with open(PATH_FILE_JSON, 'w') as f:
			json.dump(data[15000:], f, ensure_ascii=False, indent='\t')
	
	elif mode == 'test':
		with open(PATH_FILE_JSON, 'w') as f:
			json.dump(data, f, ensure_ascii=False, indent='\t')

def main():
	txt_to_json(source = PATH_FILE_TRAIN_SOURCE, output = PATH_FILE_TRAIN_OUTPUT, mode = 'train')
	txt_to_json(source = PATH_FILE_TEST_SOURCE, output = PATH_FILE_TEST_OUTPUT, mode = 'test')

if __name__ == "__main__":
	main()