#!/bin/bash

MODEL_PATH=$1
MODEL_BASENAME=$(basename $MODEL_PATH)

for task in {"winoground_filtered","vqa_filtered"}; do
	if [ $task = "winoground_filtered" ]; then
		image_src="facebook/winoground"
	elif [ $task = "vqa_filtered" ]; then
		image_src="HuggingFaceM4/VQAv2"
	fi

	python -m lm_eval --model hf \
		--model_args pretrained=$MODEL_PATH,backend="causal" \
		--tasks $task \
		--device cuda:0 \
		--batch_size 64 \
		--output_path results/${task}/${MODEL_BASENAME}/${task}_results.json \
		--image_src $image_src \
		--log_samples
done

# Use `--model hf-mlm` and `--model_args pretrained=$MODEL_PATH,backend="mlm"` if using a custom masked LM.
# Add `--trust_remote_code` if you need to load custom config/model files.