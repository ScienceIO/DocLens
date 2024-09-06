import argparse
import os
import json
import time
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import re
import openai
import openai.error
import traceback

SECTION_DIVISIONS = ['subjective', 'objective_exam', 'objective_results', 'assessment_and_plan']

def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

def completion_with_backoff(debug=False, **kwargs):
    is_ok = False
    retry_count = 0
    debug_info_path = "debug_info.json"  # Path for debug file logging retries and errors
    while not is_ok:
        retry_count += 1
        try:
            response = openai.ChatCompletion.create(**kwargs)
            is_ok = True
        except openai.error.RateLimitError as error:
            if retry_count <= 30:
                if retry_count % 10 == 0:
                    print(f"OpenAI API retry for {retry_count} times ({error})")
                time.sleep(10)
                continue
            else:
                log_error("Rate limit exceeded", locals(), debug)
                return {}
        except openai.error.InvalidRequestError as error:
            if 'maximum context length' in error._message:
                if retry_count <= 3:
                    print(f"reduce max_tokens by 500")
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

def debug_log(message, debug=False):
    if debug:
        print(f"DEBUG: {message}")
        with open("debug_log.txt", "a") as f:
            f.write(f"{message}\n")

def log_error(message, local_vars, debug=False):
    if debug:
        error_message = f"ERROR: {message}\nLocal Variables:\n{json.dumps(local_vars, indent=4, default=str)}"
        print(error_message)
        with open("error_log.txt", "a") as f:
            f.write(error_message + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', required=True, help='filename of the system-generated outputs.')
    parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset")
    parser.add_argument("--mode", type=str, default="claim_recall", choices=['claim_recall','claim_precision','same'])
    parser.add_argument("--use_persection_claims", action="store_true", default=False, help="Generate claims for each section")
    parser.add_argument('--prompt_file', required=True, help='filename of the prompt dict .json.')
    parser.add_argument("--azure", action="store_true", default=False, help="Azure openai API")
    parser.add_argument("--max_new_tokens", type=int, default=2000, help="Max number of new tokens to generate in one step")
    parser.add_argument("--model", type=str, default='gpt-4o-2024-08-06', help="see https://platform.openai.com/docs/models/gpt-4o")
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--org_id", type:str)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    if args.azure:
        openai.api_base = os.environ.get("OPENAI_API_BASE")
        openai.api_key = args.api_key
        openai.organization = args.org_id
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"
        EVALUATOR_NAME = EVALUATOR_DEPLOY_NAME = args.model
    else:
        openai.api_base = "https://api.openai.com/v1"
        openai.api_key = args.api_key
        openai.organization = args.org_id
        EVALUATOR_NAME = args.model

    savefile = args.result_file.replace('.json', f'.{args.mode}_scores')

    if not args.use_persection_claims:
        SECTION_DIVISIONS = ['full']

    output_data = json.load(open(args.result_file, 'r'))
    debug_log(f"Loaded {len(output_data)} entries from {args.result_file}. Saving scores to {savefile.split('/')[-1]}.", args.debug)

    if os.path.exists(savefile):
        claims_score = json.load(open(savefile))
    else:
        claims_score = {}
        for section in SECTION_DIVISIONS:
            claims_score[section] = {}
            for x in output_data:
                eid_str = str(x['example_id'])
                claims_score[section][eid_str] = []

    prompt_template = json.load(open(args.prompt_file, 'r'))
    for i in range(1, len(prompt_template)-1):
        prompt_template[i]['content'] = json.dumps(prompt_template[i]['content'])

    TEXT_NAME = {
        'acibench': 'clinical_note',
        'mimic': 'radiology_report',
    }

    wrong_format_count = 0
    new_generation_count = 0
    for section in SECTION_DIVISIONS:
        if args.mode == 'claim_recall':
            text_key = 'output' if args.dataset_name != 'meqsum' else 'reference'
            subclaim_key = 'subclaims_reference'
        elif args.mode == 'claim_precision':
            text_key = 'reference' if args.dataset_name != 'meqsum' else 'output'
            subclaim_key = 'subclaims_output'
        elif args.mode == 'same':
            text_key, subclaim_key = 'output', 'reference'

        text_name = TEXT_NAME.get(args.dataset_name, "clinical_report")

        for item in output_data:
            eid_str = str(item['example_id'])
            try:
                text = remove_citations(item[text_key])
            except KeyError as e:
                log_error(f"KeyError for item ID {eid_str}: {str(e)}", locals(), args.debug)
                continue

            claims = item.get(subclaim_key, [])

            if len(claims) == 0:
                claims_score[section][eid_str] = []
                continue

            if len(text) == 0:
                claims_score[section][eid_str] = [{"claim": claim, "entailment_prediction": 0} for claim in claims]
                continue

            if len(claims_score[section][eid_str]) == len(claims):
                continue

            prompt = deepcopy(prompt_template)
            prompt[-1]['content'] = json.dumps({text_name: text, "claims": claims})

            if args.debug:
                with open(f"debug_prompts_{new_generation_count}.json", "w") as f:
                    json.dump(prompt, f, indent=4)

            response = completion_with_backoff(debug=args.debug, model=EVALUATOR_NAME, messages=prompt, max_tokens=args.max_new_tokens)

            new_generation_count += 1

            try:
                judgment_dict = json.loads(response['choices'][0]['message']['content'])
                claims_score[section][eid_str] = judgment_dict
                for cid, d in enumerate(claims_score[section][eid_str]):
                    debug_log(f"Claim {cid}: {d['claim']} -> Prediction: {d['entailment_prediction']}", args.debug)
            except Exception as e:
                log_error(f'CANNOT CONVERT TO JSON: {str(e)}', locals(), args.debug)
                if args.debug:
                    with open(f"debug_response_{new_generation_count}.json", "w") as f:
                        json.dump(response, f, indent=4)
                wrong_format_count += 1

            if new_generation_count % 5 == 0:
                debug_log('Saving results..', args.debug)
                json.dump(claims_score, open(savefile, 'w'), indent=4, sort_keys=True)

    json.dump(claims_score, open(savefile, 'w'), indent=4, sort_keys=True)

    debug_log(f"Total wrong format count: {wrong_format_count}", args.debug)
