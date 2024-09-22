SAVENAME=$1
ORG_ID=$2
API_KEY=$3
PROMPT_FILE=${4:-"citation_evaluation/prompts/general_citation_entail.json"}
DATASET_NAME=${5:-user_dataset}

# eval citations 
python citation_evaluation/eval_citation.py --result_file results/${SAVENAME}.json \
    --dataset_name $DATASET_NAME --split_method citation \
    --prompt_file $PROMPT_FILE \
    --org_id $ORG_ID \
    --api_key $API_KEY \
    --debug

python aggregate_scores.py --result_file results/${SAVENAME}.json --dataset_name $DATASET_NAME \
    --eval_citations --eval_model GPT