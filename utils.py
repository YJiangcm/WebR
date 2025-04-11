from datasketch import MinHash, MinHashLSH
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import random
import numpy as np


def get_json_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as fcc_file:
        json_list = json.load(fcc_file)
    return json_list


def load_and_merge_json(file_paths):
    merged_data = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            if isinstance(data, list):
                merged_data.extend(data)
            elif isinstance(data, dict):
                merged_data.update(data) if isinstance(merged_data, dict) else merged_data.append(data)
            else:
                raise ValueError(f"Unsupported JSON format in file: {path}")
    return merged_data


def adaptive_normal_sampling(n, size=1):
    mean = n // 2
    std_dev = n // 4
    samples = np.random.normal(loc=mean, scale=std_dev, size=size)
    samples = np.clip(samples, 1, n)
    int_samples = np.round(samples).astype(int)
    return int_samples


def section(webpage, sample_method='uniform'):
    webpage_split = [w.lstrip('#').strip() for w in webpage.split("\n\n")]
    n_section = len(webpage_split)
    if n_section == 1:
        return webpage
    if sample_method == 'uniform':
        if random.uniform(0, 1) > 0.5:
            select_n = random.sample(range(1, n_section+1), 1)[0]
        else:
            select_n = n_section
    elif sample_method == 'gaussian':
         select_n = adaptive_normal_sampling(n_section)[0]
    else:
        raise ValueError(f"Unsupported sample method '{sample_method}'. Choose 'uniform' or 'gaussian'.")
    return "\n\n".join(webpage_split[:select_n])


def filter_sepcial_pattern(doc_list):
    filtered_doc_list = []
    process_num = 0
    delete_num = 0
    for i, doc in enumerate(doc_list):
        if not doc['request'] or not doc['response'] \
            or doc['response'][:len("I'm sorry")] == "I'm sorry" or doc['response'][:len("I apologize")] == "I apologize":
            delete_num += 1
            continue
        else:
            filtered_doc_list.append(doc)
    print(f'Filtering: total {len(doc_list)} samples, delete {delete_num} samples, process {process_num} samples.')
    return filtered_doc_list



def self_minhash_rm(data_pool, model_name="meta-llama/Meta-Llama-3-8B-Instruct", threshold=0.7, num_perm=128):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    minhashes = {}

    for i, doc in enumerate(tqdm(data_pool, desc="Create MinHash")):
        m = MinHash(num_perm=num_perm)
        word_list = [str(idx) for idx in tokenizer(doc['request'])['input_ids']]
        
        for word in word_list:
            m.update(word.encode('utf8'))
            
        minhashes[i] = m
        lsh.insert(f"doc_{i}", m)

    unique_documents = set()
    remove_documents = set()

    for i, mh in tqdm(minhashes.items(), desc="MinHash LSH"):
        result = lsh.query(mh)
        if result:
            representative = sorted(result, key=lambda x: int(x.split('_')[1]))
            unique_documents.add(representative[0])
            for r in representative[1:]:
                remove_documents.add(int(r.split('_')[1]))

    remaining_doc_list = [data_pool[idx] for idx in range(len(data_pool)) if idx not in remove_documents]

    print("delete: {} samples, remain ratio: {}%".format(len(data_pool)-len(remaining_doc_list), round(len(remaining_doc_list)/len(data_pool)*100, 2)))

    return remaining_doc_list