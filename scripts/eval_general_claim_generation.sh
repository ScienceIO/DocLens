SAVENAME=$1
REFERENCE=$2
ORG_ID=$3
API_KEY=$4
PROMPT_FILE=${5:-"claim_evaluation/prompts/general_subclaim_generation.json"}



# generate reference subclaims
python claim_evaluation/generate_subclaims.py --eval_file data/${REFERENCE}.json \
    --result_file results/${SAVENAME}.json \
    --mode reference_claims \
    --prompt_file $PROMPT_FILE \
    --org_id $ORG_ID \
    --api_key $API_KEY \

# generate output subclaims
python claim_evaluation/generate_subclaims.py --eval_file data/${REFERENCE}.json \
    --result_file results/${SAVENAME}.json \
    --mode output_claims \
    --prompt_file $PROMPT_FILE \
    --org_id $ORG_ID \
    --api_key $API_KEY \



