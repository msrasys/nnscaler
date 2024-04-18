import json
import argparse

parser = argparse.ArgumentParser(description='GPT Train')
parser.add_argument('--save_folder_tp',
                    type=str,
                    default='tp_data',
                    help='set the save folder for tp')
parser.add_argument('--save_folder_pp',
                    type=str,
                    default='pp_data',
                    help='set the save folder for pp')
parser.add_argument('--suffix',
                    type=str,
                    default='_nar',
                    help='set the save folder for w/o adaptive_recom')
parser.add_argument('--pp',
                    action='store_true',
                    help='for pipeline number analysis')
args = parser.parse_args()
import pandas as pd

folders = [args.save_folder_tp, args.save_folder_tp + args.suffix]
if args.pp:
    folders = [args.save_folder_pp, args.save_folder_pp + args.suffix]

model_setting_list = ['760M', '1.3B'] if args.pp else ['350M', '760M', '1.3B']
gpus = {'350M': 1, '760M': 2, '1.3B': 4, '2.6B': 8, '6.7B': 16}
recompute_list = ['True']
batch_size_list = [1, 2, 4, 8, 16, 32]

for recompute in recompute_list:
    table = {}
    for model_setting in model_setting_list:
        for batch_size in batch_size_list:
            table[batch_size] = {}
            for index, folder in enumerate(folders):
                fname = './' + folder + '/gpt3-' + model_setting + '-' + str(
                    gpus[model_setting]) + 'gpu-' + str(
                        batch_size) + 'batch_size'
                backup_fname = fname + '-backup.json'

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
                    estimated_time = -1
                    estimated_memory = -1
                    compile_time = -1

                if index == 0:
                    table[batch_size][
                        'est time w/ adapt_recom /s'] = estimated_time
                else:
                    table[batch_size][
                        'est time w/o adapt_recom /s'] = estimated_time
                if index == 0:
                    table[batch_size][
                        'compile time w/ adapt_recom /s'] = compile_time
                else:
                    table[batch_size][
                        'compile time w/o adapt_recom /s'] = compile_time
            table[batch_size]['gain/%'] = (
                table[batch_size]['est time w/o adapt_recom /s'] -
                table[batch_size]['est time w/ adapt_recom /s']
            ) / table[batch_size]['est time w/o adapt_recom /s'] * 100
            pdTable = pd.DataFrame(table).round(2).T
        print(model_setting)
        print(pdTable.to_markdown())
