import json
import argparse
import random
from queue import Queue
import threading
import time
from openai import OpenAI
from utils import get_json_list, section


#############################################################################################################################################################
author_template = """[Text]
{web}


[Instruction]
The text above is from an English webpage. According to the text, please infer the author's profile (within 30 words)."""

#############################################################################################################################################################
WI_all_template = """[Text]
{web}


[Author of the Text]
{author}


[Instruction]
The text above is from an English webpage. Imagine that you are a user of an AI assistant, please provide a rewrite {word} specifically designed based on the text content, to create a new version of the text. You can ask for the rewrite to follow constraints including word/sentence/paragraph length, style, format, structure, etc. You should also follow the below rules:
    - The rewrite {word} should strictly follow the profile of the author.
    - The rewrite {word} should be based on the above text, rather than an isolated instruction.
    - The constraints should be detailed and specific.
    - Output only the {word}.
    - Do **not** directly use the keyword 'rewrite' and 'new version' in the generated {word}.
    - Make sure the generated {word} is within {len_limit} words."""

#############################################################################################################################################################
WI_part_template = """[Text]
{web}


[Author of the Text]
{author}


[Instruction]
The text above is from an English webpage. Imagine that you are a user of an AI assistant, please provide a rewrite {word} specifically designed based on the text content, to create a new version of the text focusing on a specific part of information, rather than global information, in the given text above. You can ask for the rewrite to follow constraints including word/sentence/paragraph length, style, format, structure, etc. You should also follow the below rules:
    - The rewrite {word} should strictly follow the profile of the author.
    - The rewrite {word} should be based on the above text, rather than an isolated instruction.
    - The constraints should be detailed and specific.
    - Output only the {word}.
    - Do **not** directly use the keyword 'rewrite', 'new version', and 'specific part information' in the generated {word}.
    - Make sure the generated {word} is within {len_limit} words."""

#############################################################################################################################################################
WR_all_template = """[Text]
{web}


[Author of the Text]
{author}


[Instruction]
Imagine that you are a user of an AI assistant, please provide the most likely {word} to which the text above would be a great answer. You should also follow the below rules:
    - The {word} should strictly follow the profile of the author.
    - Ensure your {word} is detailed, specific (including the style, format, and structure of the text), clear, and concise.
    - Output only the {word}.
    - Make sure the generated {word} is within {len_limit} words."""

#############################################################################################################################################################
WR_part_template = """[Text]
{web}


[Author of the Text]
{author}

    
[Instruction]
Imagine that you are a user of an AI assistant, please provide the most likely {word} to which **a specific part of the text above** would be a great answer. You should also follow the below rules:
    - The {word} should strictly follow the profile of the author.
    - Ensure your {word} is detailed, specific (including the style, format, and structure of the text), clear, and concise.
    - Output only the {word}.
    - Make sure the generated {word} is within {len_limit} words."""

#############################################################################################################################################################
WI_response_template = """{web}


{request}"""

#############################################################################################################################################################
WR_refine_template = """Based on the Provided Information, please improve the Answer to the Question, so that the improved answer is of high quality and factually correct. Only output the improved answer.


[Provided Information]
{web}


[Question]
{request}


[Answer]
{answer}"""
#############################################################################################################################################################


class Crawl_thread(threading.Thread):
    def __init__(self, thread_id, queue, stage, category, called_model_name, temperature):
        threading.Thread.__init__(self)
        self.thread_id = thread_id  
        self.queue = queue
        self.stage = stage
        self.category = category
        self.called_model_name = called_model_name
        self.temperature = temperature

    def run(self):
        print('Start thread: ', self.thread_id)
        self.crawl_spider()
        print('End thread: ', self.thread_id)

    def crawl_spider(self):
        global all_get_data2
        while True:
            if self.queue.empty():
                break
            else:
                row = self.queue.get()

                # stage = ['author', 'request', 'response']
                # category = ['WI_all', 'WI_part', 'WR_all', 'WR_part', 'WR_refine']

                if self.stage == 'author':
                    gpt_input = author_template.format(web=row['webpage'])

                elif self.stage == 'request':
                    if self.category == 'WI_all':
                        gpt_input = WI_all_template.format(web=row['webpage'], author=row['author'], word=random.sample(['request', 'instruction'], 1)[0], len_limit=random.sample([50, 100, 150, 200], 1)[0])
                    elif self.category == 'WI_part':
                        gpt_input = WI_part_template.format(web=row['webpage'], author=row['author'], word=random.sample(['request', 'instruction'], 1)[0], len_limit=random.sample([50, 100, 150, 200], 1)[0])
                    elif self.category == 'WR_all':
                        gpt_input = WR_all_template.format(web=row['webpage'], author=row['author'], word=random.sample(['request', 'question', 'instruction'], 1)[0], len_limit=random.sample([50, 100, 150, 200], 1)[0])
                    elif self.category == 'WR_part':
                        gpt_input = WR_part_template.format(web=row['webpage'], author=row['author'], word=random.sample(['request', 'question', 'instruction'], 1)[0], len_limit=random.sample([50, 100, 150, 200], 1)[0])
                
                else:
                    if 'WI' in self.category:
                        gpt_input = WI_response_template.format(web=row['webpage'], request=row['request'])
                    else:
                        if 'refine' not in self.category:
                            gpt_input = row['request']
                        else:
                            gpt_input = WR_refine_template.format(web=row['webpage'], request=row['request'], answer=row['response'])

                try:
                    success= False
                    for attempt in range(5):
                        try:
                            response = client.chat.completions.create(
                                model=self.called_model_name,
                                temperature=self.temperature,
                                messages=[
                                    {"role": "system", "content": "You are a helpful assistant."},
                                    {"role": "user", "content": gpt_input},
                                ]
                            )
                        except Exception as e:
                            time.sleep(random.randint(1,30))
                            print(f"{e}")
                        else:
                            success = True
                            break
                    if success:
                        res = response.choices[0].message.content
                        if self.stage == 'author':
                            row['author'] = res
                        elif self.stage == 'request':
                            row['request'] = res
                        else:
                            row['response'] = res
                        all_get_data2.append(row)
                except:
                    pass


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', default=None, type=str)
    parser.add_argument('--base_url', default=None, type=str)
    parser.add_argument('--called_model_name', default="gpt-4o", type=str)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--n_threads', default=300, type=int)
    parser.add_argument('--data_path', default=None, type=str)
    parser.add_argument('--stage', default=None, type=str)
    parser.add_argument('--category', default=None, type=str)
    parser.add_argument('--save_name', default=None, type=str)

    args = parser.parse_args()


    # set api_key
    client=OpenAI(api_key='<KEY>')
    client.api_key = args.api_key
    client.base_url = args.base_url


    # load dataset
    dataset = get_json_list(args.data_path)
    if args.stage == 'request': # WI : WR = 2 : 1
        if args.category == 'WI_all':
            dataset = dataset[: len(dataset)//6*2]
        elif args.category == 'WI_part':
            dataset = dataset[len(dataset)//6*2: len(dataset)//6*4]
        elif args.category == 'WR_all':
            dataset = dataset[len(dataset)//6*4: len(dataset)//6*5]
        elif args.category == 'WR_part':
            dataset = dataset[len(dataset)//6*5: ]
    print(f'############################ {args.data_path} | number of sample: ', len(dataset))


    # call api
    all_get_data2 = []
    have_set = set([x['webpage'] for x in all_get_data2])
    pageQueue = Queue(len(dataset))
    for page in dataset: 
        if page['webpage'] not in have_set:
            pageQueue.put(page)
    print(pageQueue.qsize())

    crawl_threads = []
    crawl_name_list = range(args.n_threads)
    for thread_id in crawl_name_list:
        thread = Crawl_thread(thread_id, pageQueue, args.stage, args.category, args.called_model_name, args.temperature)
        time.sleep(0.5)
        thread.start()
        crawl_threads.append(thread)
    
    for thread in crawl_threads:
        thread.join()


    # save file        
    output_file = f"{args.save_name}.json"
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(all_get_data2, file, indent=4)

    if args.stage == 'author':
        web_sft = get_json_list(f"{args.save_name}.json")
        for d in web_sft:
            d['webpage'] = section(d['webpage'])

        output_file = f"{args.save_name}_section.json"
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(web_sft, file, indent=4)