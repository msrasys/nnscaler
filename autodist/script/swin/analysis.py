import json
import argparse

parser = argparse.ArgumentParser(description='Swin Train')
parser.add_argument('--save_folder',
                    type=str,
                    default='swin',
                    help='set the save folder for experiment data')
parser.add_argument('--pp',
                    action='store_true',
                    help='for pipeline number analysis')
args = parser.parse_args()
import pandas as pd

model_setting_list = ['toy', '355M', '1.8B']
gpus = {'toy': 1, '355M': 2, '1.8B': 4, '2.6B': 8, '6.7B': 16}
recompute_list = ['True']
batch_size_list = [1, 2, 4, 8, 16, 32]

for recompute in recompute_list:
    table = {}
    for model_setting in model_setting_list:
        for batch_size in batch_size_list:
            table[batch_size] = {}
            fname = './' + args.save_folder + '/swin-' + model_setting + '-' + str(
                gpus[model_setting]) + 'gpu-' + str(batch_size) + 'batch_size'
            estimated_fname = fname + '-estimate.json'
            backup_fname = fname + '-backup.json'
            real_fname = fname + '-real.json'

            try:
                with open(backup_fname, 'r') as f:
                    estimated_dict = json.load(f)
                    try:
                        tmp = estimated_dict['estimated memory']
                    except:
                        estimated_dict = estimated_dict[0]
                    estimated_time = estimated_dict['estimated time']
                    estimated_memory = estimated_dict['estimated memory'][
                        0] if args.pp else estimated_dict['estimated memory']
                    compile_time = estimated_dict['compile time']
            except:
                try:
                    with open(estimated_fname, 'r') as f:
                        estimated_dict = json.load(f)
                        estimated_time = estimated_dict['estimated time']
                        estimated_memory = estimated_dict['estimated memory'][
                            0] if args.pp else estimated_dict['estimated memory']
                        compile_time = estimated_dict['compile time']
                except:
                    estimated_time = -1
                    estimated_memory = -1
                    compile_time = -1
            try:
                with open(real_fname, 'r') as f:
                    real_dict = json.load(f)
                    real_time = real_dict['time/s']
                    real_memory = max(real_dict['memory/GB'].values())
            except:
                real_time = -1
                real_memory = -1

            table[batch_size]['estimation time/s'] = estimated_time
            table[batch_size]['runtime/s'] = real_time
            table[batch_size][
                'estimation memory/GB'] = estimated_memory if estimated_memory != -1 else -1
            table[batch_size][
                'runtime memory/GB'] = real_memory if real_memory != -1 else -1
            table[batch_size]['compile time/s'] = compile_time
            pdTable = pd.DataFrame(table).round(2).T
        print(model_setting, recompute)
        print(pdTable.to_markdown())
