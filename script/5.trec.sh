for task_name in 'trec'
do
echo "===== task: $task_name start! ====="
for seed_num in 8 13 42 50 60
do
echo "===== seed: $seed_num start! ====="
python deepbbt.py --seed $seed_num --task_name $task_name --device "cuda:1"
echo "===== seed: $seed_num finished! ====="
done
echo "===== task: $task_name finished! ====="
done