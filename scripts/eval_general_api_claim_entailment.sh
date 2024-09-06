SAVENAME=$1
REFERENCE=$2
ORG_ID=$3
API_KEY=$4
PROMPT_FILE=${5:-"claim_evaluation/prompts/general_claim_entail.json"}
DATASET_NAME=${6:-user_dataset}


# claim recall
python claim_evaluation/run_entailment_debug.py --result_file results/${SAVENAME}.claim_min1max90.json \
    --dataset_name $DATASET_NAME --mode claim_recall \
    --prompt_file $PROMPT_FILE \
    --org_id $ORG_ID \
    --api_key $API_KEY \
    --debug

# claim precision 
python claim_evaluation/run_entailment_debug.py --result_file results/${SAVENAME}.output_claim_min1max90.json \
    --dataset_name $DATASET_NAME --mode claim_precision \
    --prompt_file $PROMPT_FILE \
    --org_id $ORG_ID \
    --api_key $API_KEY \
    --debug

python aggregate_scores.py --result_file results/${SAVENAME}.json --dataset_name $DATASET_NAME \
    --eval_claim_recall --eval_claim_precision --eval_model GPT
