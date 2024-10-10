SAVENAME=$1
REFERENCE=$2
ORG_ID=$3
API_KEY=$4
PROMPT_FILE=${5:-"claim_evaluation/prompts/general_claim_entail.json"}
DATASET_NAME=${6:-user_dataset}

python aggregate_scores.py --result_file results/${SAVENAME}.json --dataset_name $DATASET_NAME \
    --eval_claim_recall --eval_claim_precision --eval_model GPT
