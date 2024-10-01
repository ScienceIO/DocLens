import argparse
import os
import json
import time
from tqdm import tqdm
from copy import deepcopy
import openai
import openai.error
import traceback
import re


SECTION_DIVISIONS = ['subjective', 'objective_exam', 'objective_results', 'assessment_and_plan']
error_count = 0  # Global error counter

max_retries = 21  # Set a maximum number of retries
max_temperature = 1.0  # Max temperature to cap retries


def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

def completion_with_backoff(debug=False, temperature=0.0, **kwargs):
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
            response = openai.ChatCompletion.create(
                temperature=temperature,
                # response_format={"type": "json_object"},
                response_format = response_format,
                timeout=timeout_seconds,  # Set a shorter timeout for the API request
                **kwargs
            )
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


def debug_log(message, debug=False):
    if debug:
        log_path = os.path.abspath("debug_log.txt")
        with open(log_path, "a") as f:
            f.write(f"{message}\n")


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
            f"Stack Trace:\n{stack_trace}\n"
            f"Local Variables:\n{json.dumps(local_vars, indent=4, default=str)}\n"
        )
        
        with open(error_log_path, "a") as f:
            f.write(f"{error_message}\n")
        print(f"ERROR: {message}. Details saved to {error_log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', required=True, help='filename of the system-generated outputs.')
    parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset")
    parser.add_argument("--mode", type=str, default="claim_recall", choices=['claim_recall','claim_precision','same'])
    parser.add_argument("--use_persection_claims", action="store_true", default=False, help="Generate claims for each section")
    parser.add_argument('--prompt_file', required=True, help='filename of the prompt dict .json.')
    parser.add_argument("--azure", action="store_true", default=False, help="Azure openai API")
    parser.add_argument("--max_new_tokens", type=int, default=4000, help="Max number of new tokens to generate in one step")
    parser.add_argument("--model", type=str, default='gpt-4o-2024-08-06', help="see https://platform.openai.com/docs/models/gpt-4o")
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--org_id", type=str)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--prompt_debug", action="store_true", default=False, help="Enable debug mode")

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

    savefile = os.path.abspath(args.result_file.replace('.json', f'.{args.mode}_scores'))
    print(f"Results will be saved to: {savefile}")

    if not args.use_persection_claims:
        SECTION_DIVISIONS = ['full']

    output_data = json.load(open(args.result_file, 'r'))
    if args.debug:
        debug_log(f"Loaded {len(output_data)} entries from {args.result_file}.", args.debug)

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
            # text_key = 'reference' if args.dataset_name != 'meqsum' else 'output'
            text_key = 'output'
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

            if args.prompt_debug:
                with open(f"debug_prompts_{new_generation_count}.json", "w") as f:
                    json.dump(prompt, f, indent=4)


            retry_attempts = 0
            success = False

            initial_temperature = 0.0
            while retry_attempts < max_retries and not success:
                try:
                    # Call the API with the current temperature
                    response = completion_with_backoff(debug=args.debug, temperature=initial_temperature, model=EVALUATOR_NAME, messages=prompt, max_tokens=args.max_new_tokens)
                    # In case it fails, start with a high enough temperature so that the outcomes will be different
                    initial_temperature = 0.5
                    new_generation_count += 1

                    to_load = response['choices'][0]['message']['content']

                    if to_load.startswith("```json"):
                        to_load = to_load[7:]  # Removes the "```json\n" part
                    if to_load.endswith("```"):
                        to_load = to_load[:-3]  # Removes the ending "```"

                    judgment_dict = json.loads(to_load)
                    judgment_dict = judgment_dict['entailments']
                    if isinstance(judgment_dict, dict):
                        judgment_dict = [judgment_dict]

                    claims_score[section][eid_str] = judgment_dict
                    for cid, d in enumerate(claims_score[section][eid_str]):
                        print(f"{d['entailment_prediction']} Claim {cid}: {d['claim']}")
                        debug_log(f"Claim {cid}: {d['claim']} -> Prediction: {d['entailment_prediction']}", args.debug)
                    success = True  # Set success to True to break out of retry loop
                    #Reset the temperature for the next step
                    initial_temperature = 0.0
                    print('Success')
                except Exception as e:
                    # Increment retry count and temperature for the next retry
                    retry_attempts += 1
                    initial_temperature = min(initial_temperature + 0.1, max_temperature)  # Increment temperature by 0.1
                    print(f"Retrying with temperature: {initial_temperature}")

                    if retry_attempts >= max_retries:
                        # Log error only if all attempts have failed
                        log_error(f'CANNOT CONVERT TO JSON after {max_retries} attempts: {str(e)}', locals(), args.debug)
                        #Reset the temperature for the next step
                        initial_temperature = 0.0
                        break

            if success and args.debug and retry_attempts % 21 == 0:
                with open(f"debug_response_{new_generation_count}.json", "w") as f:
                    json.dump(response, f, indent=4)

            # Saving intermediate results
            if new_generation_count % 5 == 0:
                debug_log('Saving intermediate results..', args.debug)
                json.dump(claims_score, open(savefile, 'w'), indent=4, sort_keys=True)

        json.dump(claims_score, open(savefile, 'w'), indent=4, sort_keys=True)


    if args.debug:
        debug_log(f"Total wrong format count: {wrong_format_count}", args.debug)




