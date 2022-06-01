
PATH_TO_STUDENT_MODEL_CONFIG="TODO"

job_name=6T6

# the sample uses a server with 8 GPUs
deepspeed --num_nodes 1 \
    general_distill.py \
        --job_name ${job_name} \
        --deepspeed \
        --deepspeed_config ds_lamb_config.json \
        --student_model_class SkipBert \
        --num_train_epochs 10 \
        --pregenerated_data /tmp/corpus \
        --teacher bert-base-uncased \
        --student_model ${PATH_TO_STUDENT_MODEL_CONFIG} \
        --output_dir output_${job_name} \
        --num_hidden_layers 12 \
        --num_full_hidden_layers 6 \
        --num_masked_layers_teacher 6