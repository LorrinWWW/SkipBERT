
rm *.memmap

student_model=../model-6-2-new

model_type_student=SkipBert

num_layers_student=8
num_full_hidden_layers_student=2
num_masked_layers_teacher=0
num_masked_last_layers_teacher=0

TASK_NAME=CoLA
task_name=cola
eval_steps=50
epochs_no_cls=0
batch_size=32
alpha=1
lr=5
epoch=20

beta=0.1

teacher_model=./teachers/cola_teacher/

OUTPUT_DIR=./model/${TASK_NAME}/
LOG_OUTPUT_PATH=${OUTPUT_DIR}log_${student_model}.txt

mkdir -p ${OUTPUT_DIR}
        
CUDA_VISIBLE_DEVICES=1 \
python task_distill.py \
    --train_batch_size ${batch_size} \
    --eval_batch_size 128 \
    --data_dir ./data/${TASK_NAME}/ \
    --teacher_model ${teacher_model} \
    --student_model ${student_model} \
    --student_model_tokenizer ${student_model} \
    --model_type_student ${model_type_student} \
    --task_name ${TASK_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --num_layers_student ${num_layers_student} \
    --num_full_hidden_layers_student ${num_full_hidden_layers_student} \
    --num_masked_layers_teacher ${num_masked_layers_teacher} \
    --num_masked_last_layers_teacher ${num_masked_last_layers_teacher} \
    --att_layer_maps 3 7 \
    --hid_layer_maps 9 10 11 \
    --epochs_no_cls ${epochs_no_cls} \
    --epochs_no_eval 0 \
    --use_embedding false \
    --use_att true \
    --use_rep true \
    --use_logits true \
    --learning_rate ${lr}e-5 \
    --num_train_epochs $((${epoch} + ${epochs_no_cls})) \
    --alpha ${alpha} \
    --beta ${beta} \
    --eval_step ${eval_steps} \
    --freeze_lower_layers true \
    --do_fit false \
    --share_param false