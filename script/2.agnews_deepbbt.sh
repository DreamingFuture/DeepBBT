cuda=2
CUDA_VISIBLE_DEVICES=$cuda python deepbbt.py \
    --seed 8 \
    --task_name agnews

CUDA_VISIBLE_DEVICES=$cuda python deepbbt.py \
    --seed 13 \
    --task_name agnews

CUDA_VISIBLE_DEVICES=$cuda python deepbbt.py \
    --seed 42 \
    --task_name agnews

CUDA_VISIBLE_DEVICES=$cuda python deepbbt.py \
    --seed 50 \
    --task_name agnews

CUDA_VISIBLE_DEVICES=$cuda python deepbbt.py \
    --seed 60 \
    --task_name agnews