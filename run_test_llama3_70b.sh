#!/bin/bash


called_model_name="llama3-70b"
data_name="web_en_1k"
tokenizer_model_name="Meta-Llama-3-8B-Instruct"


run_api() {
    local stage=$1
    local data_path=$2
    local category=$3
    local save_name=$4

    echo "Running stage: $stage, category: ${category:-"none"}"
    python create_data_llama3_70b.py \
        --data_path="$data_path" \
        --stage="$stage" \
        ${category:+--category="$category"} \
        --save_name="$save_name"

    if [ $? -ne 0 ]; then
        echo "Error: Failed to run stage $stage with category ${category:-"none"}"
        exit 1
    fi
}


# Create data

## Stage 1: Author
run_api "author" "${data_name}.json" "" "${data_name}_${called_model_name}"


## Stage 2 and 3: Request and Response
categories=( "WI_all" "WI_part" "WR_all" "WR_part" )
for category in "${categories[@]}"; do
    run_api "request" "${data_name}_${called_model_name}_section.json" "$category" "${data_name}_${called_model_name}_section_$category"
    run_api "response" "${data_name}_${called_model_name}_section_${category}.json" "$category" "${data_name}_${called_model_name}_section_$category"
done


## Additional refine responses
refine_pairs=( "WR_all" "WR_part" )
for refine in "${refine_pairs[@]}"; do
    run_api "response" "${data_name}_${called_model_name}_section_${refine}.json" "WR_refine" "${data_name}_${called_model_name}_section_${refine}_refine"
done


# Post-process the results
python post_process.py \
    --WI_data_paths "${data_name}_${called_model_name}_section_WI_all.json" "${data_name}_${called_model_name}_section_WI_part.json" \
    --WR_data_paths "${data_name}_${called_model_name}_section_WR_all_refine.json" "${data_name}_${called_model_name}_section_WR_part_refine.json" \
    --tokenizer_model_name $tokenizer_model_name \
    --save_name $data_name

if [ $? -ne 0 ]; then
    echo "Error: Post-processing failed"
    exit 1
fi

echo "Script completed successfully."