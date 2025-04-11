import argparse
import random
import pyarrow.parquet as pq
import pyarrow
import pandas as pd
from utils import load_and_merge_json, filter_sepcial_pattern, self_minhash_rm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--WI_data_paths', nargs='+', type=str, help='List of WI_data_paths')
    parser.add_argument('--WR_data_paths', nargs='+', type=str, help='List of WR_data_paths')
    parser.add_argument('--tokenizer_model_name', default="meta-llama/Meta-Llama-3-8B-Instruct", type=str)
    parser.add_argument('--save_num', default=100000, type=int)
    parser.add_argument('--save_name', default=None, type=str)

    args = parser.parse_args()


    # load data
    long_to_long_data = load_and_merge_json(args.WI_data_paths)
    short_to_long_data = load_and_merge_json(args.WR_data_paths)


    # filtering
    long_to_long_data_final = filter_sepcial_pattern(long_to_long_data)
    long_to_long_data_final = self_minhash_rm(long_to_long_data_final, model_name=args.tokenizer_model_name, threshold=0.7, num_perm=128)

    short_to_long_data_final = filter_sepcial_pattern(short_to_long_data)
    short_to_long_data_final = self_minhash_rm(short_to_long_data_final, model_name=args.tokenizer_model_name, threshold=0.7, num_perm=128)


    # WI : WR = 2 : 1
    random.shuffle(long_to_long_data_final)
    random.shuffle(short_to_long_data_final)

    long_to_long_data = long_to_long_data_final[: 100000//3*2]
    web_set = set([d['webpage'] for d in long_to_long_data])

    short_to_long_data = []
    for d in short_to_long_data_final:
        if d['webpage'] not in web_set:
            short_to_long_data.append(d)
            web_set.add(d['webpage'])
        if len(short_to_long_data) == 100000//3:
            break


    # concat
    all_data = []
    for i, data in enumerate(long_to_long_data):
        if random.uniform(0, 1) > 0.5:
            content = [{'content': "{}\n\n{}". format(data['request'], data['webpage']), 'role': 'user'}, {'content': data['response'], 'role': 'assistant'}]
        else:
            content = [{'content': "{}\n\n{}". format(data['webpage'], data['request']), 'role': 'user'}, {'content': data['response'], 'role': 'assistant'}]
        all_data.append({
                        'prompt_id': 'web_data_' + str(i),
                        'messages': content,
        })

    for j, data in enumerate(short_to_long_data):
        if 'improved answer' in data['response'].lower():
            content = [{'content': data['request'], 'role': 'user'}, {'content': "\n\n".join(data['response'].split('\n\n')[1:]).strip('-').strip('\n').strip(), 'role': 'assistant'}]
        else:
            content = [{'content': data['request'], 'role': 'user'}, {'content': data['response'], 'role': 'assistant'}]
        all_data.append({
                        'prompt_id': 'web_data_' + str(i+j+1),
                        'messages': content,
        })


    # save file
    # import json
    # output_file = f"{args.save_name}.json"
    # with open(output_file, 'w', encoding='utf-8') as file:
    #     json.dump(all_data[: args.save_num], file, ensure_ascii=False, indent=4)


    df = pd.DataFrame(all_data[: args.save_num])
    print("Ready to save {} samples.".format(len(df)))
    pq.write_table(table=pyarrow.Table.from_pandas(df), where=f"{args.save_name}.parquet")