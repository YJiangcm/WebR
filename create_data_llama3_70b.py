from vllm import LLM, SamplingParams
import json
import argparse
import random
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="meta-llama/Meta-Llama-3-70B-Instruct", type=str)
    parser.add_argument('--data_path', default=None, type=str)
    parser.add_argument('--tensor_parallel_size', default=4, type=int)
    parser.add_argument('--max_tokens', default=4096, type=int)
    parser.add_argument('--stage', default=None, type=str)
    parser.add_argument('--category', default=None, type=str)
    parser.add_argument('--save_name', default=None, type=str)

    args = parser.parse_args()

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

    # Create an LLM
    llm = LLM(model=f"{args.model_path}", tensor_parallel_size=args.tensor_parallel_size)
    tokenizer = llm.get_tokenizer()
    if 'llama-3' in args.model_path.lower():
        pattern = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    else:
        raise NameError
    
    # Create a sampling params object
    sampling_params = SamplingParams(
                                     n=1,
                                     temperature=0.6,
                                     top_p=0.9,
                                     max_tokens=args.max_tokens,
                                     stop_token_ids=stop_token_ids,
                                     )

    # Create data
    prompts = []
    for idx in range(len(dataset)):

        if args.stage == 'author':
            llm_input = author_template.format(web=dataset[idx]['webpage'])

        elif args.stage == 'request':
            if args.category == 'WI_all':
                llm_input = WI_all_template.format(web=dataset[idx]['webpage'], author=dataset[idx]['author'], word=random.sample(['request', 'instruction'], 1)[0], len_limit=random.sample([50, 100, 150, 200], 1)[0])
            elif args.category == 'WI_part':
                llm_input = WI_part_template.format(web=dataset[idx]['webpage'], author=dataset[idx]['author'], word=random.sample(['request', 'instruction'], 1)[0], len_limit=random.sample([50, 100, 150, 200], 1)[0])
            elif args.category == 'WR_all':
                llm_input = WR_all_template.format(web=dataset[idx]['webpage'], author=dataset[idx]['author'], word=random.sample(['request', 'question', 'instruction'], 1)[0], len_limit=random.sample([50, 100, 150, 200], 1)[0])
            elif args.category == 'WR_part':
                llm_input = WR_part_template.format(web=dataset[idx]['webpage'], author=dataset[idx]['author'], word=random.sample(['request', 'question', 'instruction'], 1)[0], len_limit=random.sample([50, 100, 150, 200], 1)[0])
        
        else:
            if 'WI' in args.category:
                llm_input = WI_response_template.format(web=dataset[idx]['webpage'], request=dataset[idx]['request'])
            else:
                if 'refine' not in args.category:
                    llm_input = dataset[idx]['request']
                else:
                    llm_input = WR_refine_template.format(web=dataset[idx]['webpage'], request=dataset[idx]['request'], answer=dataset[idx]['response'])

        prompts.append(pattern.format(llm_input))

    outputs = llm.generate(prompts, sampling_params)


    # save file
    saved_output = []
    for idx in range(len(outputs)):

        if args.stage == 'author':
            saved_output.append({
                                'webpage': dataset[idx]['webpage'],
                                'author': outputs[idx].outputs[0].text,
                                })
            
        elif args.stage == 'request':
            request = outputs[idx].outputs[0].text
            if request[:len('Here is')] == 'Here is' or request[:len("Here's")] == "Here's":
                request = "\n\n".join(request.split("\n\n")[1:])
            request = request.strip("\"")
            saved_output.append({
                                'webpage': dataset[idx]['webpage'],
                                'author': dataset[idx]['author'],
                                'request': request,
                                })

        else:
            response = outputs[idx].outputs[0].text
            if response[:len('Here is')] == 'Here is' or response[:len("Here's")] == "Here's":
                response = "\n\n".join(response.split("\n\n")[1:])
            response = response.strip("\"")
            saved_output.append({
                                'webpage': dataset[idx]['webpage'],
                                'author': dataset[idx]['author'],
                                'request': dataset[idx]['request'],
                                'response': response,
                                })
            
    output_file = f"{args.save_name}.json"
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(saved_output, file, indent=4)

    if args.stage == 'author':
        web_sft = get_json_list(f"{args.save_name}.json")
        for d in web_sft:
            d['webpage'] = section(d['webpage'])

        output_file = f"{args.save_name}_section.json"
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(web_sft, file, indent=4)