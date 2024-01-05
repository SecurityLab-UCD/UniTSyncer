import os
import jsonlines
import fire
import json
from sklearn.model_selection import train_test_split

#save data
def save_data(data, filename):
        with open(filename, 'w') as file:
            for test_id, code, test in data:
                file.write(json.dumps({'test_id': test_id, 'code': code, 'test': test}) + '\n')

def main():
    source_codes = []
    source_folder = './training/source'
    # output_folder = './training/data'
    for filename in os.listdir(source_folder):
        if filename.endswith('success.jsonl'):
            file_path = os.path.join(source_folder, filename)
            with jsonlines.open(file_path) as reader:
                for _,obj in enumerate(reader):
                    source_codes.append((obj['test_id'],obj['code'],obj['test']))
    # divided the datasets
    train_data, temp_data = train_test_split(source_codes, test_size=0.4, random_state=42)
    valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    save_data(train_data, './training/data/train_dataset.jsonl')
    save_data(valid_data, './training/data/valid_dataset.jsonl')
    save_data(test_data, './training/data/test_dataset.jsonl')
if __name__ == "__main__":
    fire.Fire(main)



