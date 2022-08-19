cuda=3
CUDA_VISIBLE_DEVICES=$cuda python deepbbt.py \
    --seed 8 \
    --task_name mrpc

CUDA_VISIBLE_DEVICES=$cuda python deepbbt.py \
    --seed 13 \
    --task_name mrpc

CUDA_VISIBLE_DEVICES=$cuda python deepbbt.py \
    --seed 42 \
    --task_name mrpc

CUDA_VISIBLE_DEVICES=$cuda python deepbbt.py \
    --seed 50 \
    --task_name mrpc

CUDA_VISIBLE_DEVICES=$cuda python deepbbt.py \
    --seed 60 \
    --task_name mrpc