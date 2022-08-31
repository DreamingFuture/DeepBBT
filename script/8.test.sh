# 'trec'
for task_name in 'agnews' 'mrpc' 'sst2' 'snli' 'yelpp'
do
echo "===== task: $task_name start! ====="
python deep_test.py  --task_name $task_name --device "cuda:2" --batch_size 4
echo "===== task: $task_name finished! ====="
done