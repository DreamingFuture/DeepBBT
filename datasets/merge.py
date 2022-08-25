import os

seeds = [8, 13, 42, 50, 60]
task_list = ['sst2', 'mrpc', 'agnews', 'trec', 'yelpp', 'snli']
task_name_dict = {
    'sst2': 'SST-2',
    'agnews': 'AGNews',
    'yelpp': 'Yelp',
    'mrpc': 'MRPC',
    'snli': 'SNLI',
    'trec': 'TREC'
}

# for task_name in task_list:
    
#     for seed in seeds:
#         res = []
#         for index in seeds:
#             content = ""
#             if seed != index:
#                 path = os.path.join('./datasets', task_name_dict[task_name], str(index), 'train.tsv')
#                 with open(path, 'r', encoding='utf-8') as fr:
#                     content = fr.readline()
#                     while content:
#                         res.append(content.split('\t')[0])
#                         content = fr.readline()
                    
#             path = os.path.join('./datasets', task_name_dict[task_name], str(index), 'dev.tsv')
#             with open(path, 'r', encoding='utf-8') as fr:
#                 content = fr.readline()
#                 while content:
#                     res.append(content.split('\t')[0])
#                     content = fr.readline()
#         print(task_name, seed, len(res), len(set(res)))

for task_name in task_list:
    for seed in seeds:
        res = []
        
        path = os.path.join('./datasets', task_name_dict[task_name], str(seed), 'test.tsv')
        with open(path, 'w', encoding='utf-8') as f:
            
            for index in seeds:
                content = ""
                
                if seed != index:
                    path = os.path.join('./datasets', task_name_dict[task_name], str(index), 'train.tsv')
                    with open(path, 'r', encoding='utf-8') as fr:
                        content = fr.readline()
                        while content:
                            if content.split('\t')[0] not in res:
                                res.append(content.split('\t')[0])
                                f.write(content)                      
                            content = fr.readline()
                
                    path = os.path.join('./datasets', task_name_dict[task_name], str(index), 'dev.tsv')
                    with open(path, 'r', encoding='utf-8') as fr:
                        content = fr.readline()
                        while content:
                            if content.split('\t')[0] not in res:
                                res.append(content.split('\t')[0])
                                f.write(content)                      
                            content = fr.readline()

for task_name in task_list:
    for seed in seeds:
        res = []
        path = os.path.join('./datasets', task_name_dict[task_name], str(seed), 'test.tsv')
        with open(path, 'r', encoding='utf-8') as fr:
            content = fr.readline()
            while content:
                res.append(content.split('\t')[0])
                content = fr.readline()
        print(task_name, seed, len(res), len(set(res)))
