import argparse
import os
import json
import time
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import re
import traceback
import sys

import openai
import openai.error

import nltk
nltk.download('punkt_tab')
from nltk import sent_tokenize

SECTION_DIVISIONS = ['subjective', 'objective_exam', 'objective_results', 'assessment_and_plan']

# Add the global counter
debug_log_counter = 0

def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

def completion_with_backoff(debug=False, temperature=0.0, **kwargs):
    global debug_log_counter  # Use the global counter
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "entailment_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "entailments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "claim": {
                                    "type": "string"
                                },
                                "entailment_prediction": {
                                    "type": "integer"
                                },
                                "explanation": {
                                    "type": "string"
                                }
                            },
                            "required": ["claim", "entailment_prediction", "explanation"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["entailments"],
                "additionalProperties": False
            }
        }
    }
    is_ok = False
    retry_count = 0
    max_retries = 21  # You can increase this value as needed
    timeout_seconds = 60  # Adjust the timeout duration for each request

    while not is_ok and retry_count < max_retries:
        retry_count += 1
        try:
            if debug and debug_log_counter < 1:
                save_prompt_to_prompt_file(kwargs)  # Save prompts to a separate file
                # Log local variables before the API call
                log_debug_info("Before API call", locals(), debug)

            response = openai.ChatCompletion.create(
                temperature=temperature,
                response_format=response_format,
                timeout=timeout_seconds,  # Set a shorter timeout for the API request
                **kwargs
            )

            if debug and debug_log_counter < 1:
                # Log local variables after the API call
                log_debug_info("After API call", locals(), debug)

            is_ok = True
        except openai.error.Timeout as error:
            print(f"Request timed out on attempt {retry_count}/{max_retries}. Retrying...")
            if retry_count >= max_retries:
                log_error("Request timed out after max retries.", locals(), debug)
                return {}
            time.sleep(10)  # Backoff time before retrying
        except openai.error.RateLimitError:
            if retry_count <= max_retries:
                time.sleep(10)
                continue
            else:
                log_error("Rate limit exceeded", locals(), debug)
                return {}
        except openai.error.InvalidRequestError as error:
            if 'maximum context length' in error._message:
                if retry_count <= 3:
                    kwargs['max_tokens'] = kwargs['max_tokens'] - 500
                    continue
                else:
                    log_error("Invalid request error", locals(), debug)
                    return {}
            else:
                log_error("API error", locals(), debug)
                return {}
        except Exception as error:
            log_error("Unexpected error", locals(), debug)
            return {}

    return response

def save_prompt_to_prompt_file(prompt_data):
    global debug_log_counter
    if debug_log_counter < 1:
        prompt_log_path = os.path.abspath("prompts_log.txt")
        print('Saving prompts to ', prompt_log_path)
        with open(prompt_log_path, "a") as f:
            f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(json.dumps(prompt_data, indent=4, default=str))
            f.write('\n---\n')
        # Increment the counter after logging
        debug_log_counter += 1

def log_debug_info(message, local_vars, debug=False):
    global debug_log_counter
    if debug and debug_log_counter < 1:
        debug_log_path = os.path.abspath("debug_log.txt")
        separator = "====================="

        # Optionally, remove large or sensitive variables
        # For example, exclude 'response' or 'kwargs' if they are too big
        # local_vars = {k: v for k, v in local_vars.items() if k not in ['response', 'kwargs']}

        debug_message = (
            f"{separator}\nDEBUG: {message}\n"
            f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Local Variables:\n{json.dumps(local_vars, indent=4, default=str)}\n"
        )

        with open(debug_log_path, "a") as f:
            f.write(f"{debug_message}\n")
        # Increment the counter after logging
        debug_log_counter += 1

def log_error(message, local_vars, debug=False):
    global error_count
    if debug:
        error_count += 1  # Increment the error counter
        error_log_path = os.path.abspath("error_log.txt")
        separator = "====================="

        # Remove 'output_data' from local_vars if it exists
        if 'output_data' in local_vars:
            local_vars = {k: v for k, v in local_vars.items() if k != 'output_data'}

        # Capture the stack trace
        stack_trace = traceback.format_exc()

        # Build the error message
        error_message = (
            f"{separator}\nError number {error_count}:\n"
            f"ERROR: {message}\n"
            f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Stack Trace:\n{stack_trace}\n"
            f"Local Variables:\n{json.dumps(local_vars, indent=4, default=str)}\n"
        )

        with open(error_log_path, "a") as f:
            f.write(f"{error_message}\n")
        print(f"ERROR: {message}. Details saved to {error_log_path}")

def log_info(message):
    info_log_path = os.path.abspath("info_log.txt")
    with open(info_log_path, "a") as f:
        f.write(f"{message}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--result_file', required=True, help='filename of the system-generated outputs.')
    parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset")

    # evaluation setting
    parser.add_argument("--split_method", type=str, choices=['sent', 'citation'], help="Split the generation output by sent/citation idx")
    parser.add_argument("--max_citation_num", type=int, default=90)
    parser.add_argument("--get_persection_score", action="store_true", default=False, help="Compute the scores for each section")

    # evaluation model
    parser.add_argument('--prompt_file', required=True, help='filename of the prompt dict .json.')
    parser.add_argument("--azure", action="store_true", default=False, help="Azure openai API")
    parser.add_argument("--max_new_tokens", type=int, default=2000, help="Max number of new tokens to generate in one step")
    parser.add_argument("--model", type=str, default='gpt-4o-2024-08-06', help="see https://platform.openai.com/docs/models/gpt-4o")

    # parser.add_argument("--api_key", type=str)
    # parser.add_argument("--org_id", type=str)
    parser.add_argument("--debug", action="store_true", default=False, help="If set, save prompts sent to OpenAI API to a debug file.")

    args = parser.parse_args()

    result_file, dataset_name, split_method, max_citation_num, prompt_file, max_new_tokens = args.result_file, args.dataset_name, args.split_method, args.max_citation_num, args.prompt_file, args.max_new_tokens
    savefile = result_file.replace('.json', '.citations.score')

    # API setup
    if args.azure:
        openai.api_base = os.environ.get("OPENAI_API_BASE")
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"
        EVALUATOR_NAME = EVALUATOR_DEPLOY_NAME = args.model
        # EVALUATOR_NAME = EVALUATOR_DEPLOY_NAME = "gpt-35-turbo"
    else:
        if not args.api_key or not args.org_id:
            print("API key or Organization ID not provided.")
            sys.exit(1)
        openai.api_base = "https://api.openai.com/v1"
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        openai.organization = os.environ.get("OPENAI_ORG_ID")

        EVALUATOR_NAME = args.model

    if not args.get_persection_score:
        SECTION_DIVISIONS = ['full']

    output_data = json.load(open(result_file, 'r')) # a list of dicts

    print( f"Saving scores to {savefile.split('/')[-1]}..") # {section: {eid_str: [{"send_id": "", "output": "", ... "entailment_prediction": 0 or 1}, ...]} }

    if os.path.exists(savefile):
        print('Save file exists!')
        citations_score = json.load(open(savefile, 'r'))
    else:
        citations_score = {}
        for section in SECTION_DIVISIONS:
            citations_score[section] = {}
            for x in output_data:
                eid_str = str(x['example_id'])
                citations_score[section][eid_str] = []

    TEXT_NAME = {
        'acibench': {'output_sent_name': 'sentence_in_note', 'cited_input_name': 'conversational_turns'},
        'mimic': {'output_sent_name': 'sentence_in_summary', 'cited_input_name': 'sentences_in_the_radiology_report'},
        'meqsum': {'output_sent_name': 'short_question', 'cited_input_name': 'sentences_in_the_long_question'},
    }

    # run entailment
    wrong_format_count = 0
    wrong_entailment_count = 0
    sent_count = 0
    new_generation_count = 0
    error_count = 0  # Initialize error counter
    for section in SECTION_DIVISIONS:
        if args.get_persection_score:
            output_key = f'output_{section}'
            prompt_template = json.load(open(prompt_file.replace('persection', section), 'r'))
        else:
            output_key = 'output'
            prompt_template = json.load(open(prompt_file, 'r'))

        for i in range(1, len(prompt_template)-1):
            prompt_template[i]['content'] = json.dumps(prompt_template[i]['content'])

        if dataset_name in TEXT_NAME:
            output_sent_name, cited_input_name = TEXT_NAME[dataset_name]["output_sent_name"], TEXT_NAME[dataset_name]["cited_input_name"]
        else:
            output_sent_name, cited_input_name = "generated_sentence", "sentence_in_clinical_report"

        for item in output_data:
            eid_str, input_text, output_text = str(item['example_id']), item['input'], item[output_key]

            if args.debug:
                log_debug_info(f"Processing example_id: {eid_str}", locals(), args.debug)

            if output_text == "":
                # skip empty note
                citations_score[section][eid_str] = []
                continue

            # preprocess input (split output note into sents, split input_text by idx)
            if dataset_name == 'meqsum':
                # only one sent in the generation output
                sents = [output_text]
            elif split_method == 'sent':
                sents = sent_tokenize(output_text)
            elif split_method == 'citation':
                # print('output_text', output_text)
                clean_sents = re.split("[\[\d+\]]+", output_text)[:-1] # remove the last split without citations
                # print('clean_sents ', clean_sents)
                citations_list = re.findall("[\[\d+\]]+", output_text)
                # print('citations_list ', citations_list)
                sents = [s+c for s,c in zip(clean_sents, citations_list)]
                # print('sents', sents)
                if len(sents) == 1:
                    log_info('Citation not found')
                    wrong_format_count += 1
                    sent_count += 1
                    citations_score[section][eid_str] = [{
                        "sent_id": 0,
                        "output": "",
                        "citations": [],
                        "cited_sents": [],
                        "entailment_prediction": 0,
                        "explanation": "",
                        "provenance": [],
                    }]

                    continue

            sents = [" ".join(s.split()) for s in sents] # output sents w/ citations
            target_sents = [remove_citations(sent) for sent in sents]

            # split input text by citations
            input_sents = re.split("\[\d+\]", input_text)[1:] # the sent is after its citation idx
            citations_input = re.findall("\[\d+\]", input_text)
            print(citations_input)
            input_sents = [" ".join(s.split()) for s in input_sents]
            docs = {int(citation[1:-1]): sent for sent, citation in zip(input_sents, citations_input)}

            # run entailment
            sent_count += len(sents)
            new_gen_flag = False

            if len(citations_score[section][eid_str]) < len(sents):
                citations_score[section][eid_str] = [{} for _ in sents]

            for sent_id, sent in enumerate(sents):
                if "entailment_prediction" in citations_score[section][eid_str][sent_id]:
                    continue

                new_gen_flag = True

                target_sent = target_sents[sent_id] # The output sent

                # Find references

                ref = [int(r[1:]) for r in re.findall(r"\[\d+", sent)] # In our setting the citation starts from 0
                ref = list(set(ref)) # there could be repeated ref

                # Instead of printing to console, write to a separate file
                message = f"{'-'*20} eid_str: {eid_str}, Sentence idx: {sent_id} {'-'*20}\n"
                message += f"For `{sent}`, find citations {ref}"
                log_info(message)

                if len(ref) == 0:
                    # No citations
                    # Reach the next citation
                    for next_sent_id in range(sent_id+1, len(sents)):
                        next_sent = sents[next_sent_id]
                        next_target_sent = target_sents[next_sent_id]
                        ref = [int(r[1:]) for r in re.findall(r"\[\d+", next_sent)]
                        if len(ref) > 0:
                            break
                    # Log the info
                    message = f"For `{sent}`, find citations {ref}"
                    log_info(message)

                if len(ref) == 0 or any([ref_id >= len(docs) for ref_id in ref]):
                    # No citations or Citations out of range
                    log_info(f"Invalid citation format: {ref}")
                    wrong_format_count += 1
                    citations_score[section][eid_str][sent_id] = {
                        "sent_id": sent_id,
                        "output": sent,
                        "citations": ref,
                        "cited_sents": [],
                        "entailment_prediction": 0,
                        "explanation": "",
                        "provenance": [],
                    }

                    continue

                ref = ref[:args.max_citation_num]

                # compute citation scores
                if dataset_name == 'acibench':
                    joint_passage = []
                    for psgs_id in ref:
                        speaker = re.findall(r"\[[a-z,\s,_]+\]", docs[psgs_id])[0][1:-1]
                        content = re.sub(r"\[[a-z,\s,_]+\] ", "", docs[psgs_id])
                        joint_passage.append({
                            "idx": str(psgs_id),
                            "speaker": speaker,
                            "content": content
                        })
                else:
                    joint_passage = []
                    for psgs_id in ref:
                        joint_passage.append({
                            "idx": str(psgs_id),
                            "content": docs[psgs_id]
                        })

                # Log the joint_passage instead of printing
                log_info(json.dumps(joint_passage, indent=4))

                prompt = deepcopy(prompt_template)
                prompt[-1]['content'] = json.dumps({
                    output_sent_name: target_sent,
                    cited_input_name: joint_passage
                })

                if args.debug:
                    log_debug_info(f"API call parameters for sentence id {sent_id}", locals(), args.debug)

                if args.azure:
                    response = completion_with_backoff(
                        engine=EVALUATOR_DEPLOY_NAME, model=EVALUATOR_NAME, messages=prompt, max_tokens=max_new_tokens, debug=args.debug
                    )
                else:
                    response = completion_with_backoff(
                        model=EVALUATOR_NAME, messages=prompt, max_tokens=max_new_tokens, debug=args.debug
                    )

                if len(response) == 0:
                    citations_score[section][eid_str][sent_id] = {
                        "sent_id": sent_id,
                        "output": sent,
                        "citations": ref,
                        "cited_sents": joint_passage,
                        "response": "",
                    }
                    log_info('No response from the evaluator model')
                    wrong_entailment_count += 1

                    continue
                else:
                    response_content = response['choices'][0]['message']['content']

                try:
                    response_dict = json.loads(response_content) # entailment_prediction, explanation, provenance
                    # Log the response_dict instead of printing
                    log_info(json.dumps(response_dict, indent=4))

                    response_dict.update({
                        "sent_id": sent_id,
                        "output": sent,
                        "citations": ref,
                        "cited_sents": joint_passage,
                        "entailment_prediction": response_dict['entailment_prediction'],
                        "explanation": response_dict['explanation'],
                        "provenance": response_dict['provenance'],
                    })

                    citations_score[section][eid_str][sent_id] = response_dict
                except:
                    wrong_entailment_count += 1
                    log_info('!'*10 + ' Cannot convert to json format ' + '!'*10)
                    log_info(response_content)
                    citations_score[section][eid_str][sent_id] = {
                        "sent_id": sent_id,
                        "output": sent,
                        "citations": ref,
                        "cited_sents": joint_passage,
                        "response": response_content,
                    }

                    if args.debug:
                        log_error("Failed to parse response JSON", locals(), args.debug)

            new_generation_count += int(new_gen_flag)
            if new_gen_flag and new_generation_count % 3 == 0:
                print('Saving results..')
                json.dump(citations_score, open(savefile, 'w'), indent=4, sort_keys=True)

    # save results
    json.dump(citations_score, open(savefile, 'w'), indent=4, sort_keys=True)

    print(f"Wrong format count: {wrong_format_count}/{sent_count}")
    print(f"Wrong entailment count: {wrong_entailment_count}/{sent_count}")
