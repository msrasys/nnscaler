import os
import time
import json
import argparse

LAYER_DOT_NUM = 6
MAX_LAYER = 64


class strategy:

    def __init__(self, Instruction: str):
        self.elements = Instruction.split(' ')
        self.id = self.elements[1]
        at_index = self.elements.index('@')
        self.selected_str = self.elements[at_index - 5:at_index]
        self.selected_str = ' '.join(self.selected_str)


def write_str(save_file, strs):
    lines = []
    for i in strs:
        lines += i
    with open(save_file, 'w') as f:
        f.writelines(lines)
    return


def get_str(args, lines, indexs, selected_strs):
    strs = []
    dot_name = {
        0: 'qvk_combined',
        1: '...qhd,...khd->...hqk',
        2: '...hqk,...khd->...qhd',
        3: 'attention/output',
        4: 'intermediate/dense',
        5: 'output/dense'
    }
    assert len(indexs) % LAYER_DOT_NUM == 0
    str_count = 0
    for dot_count, index in enumerate(indexs):

        this_strs = []
        assert 'Instruction' in lines[index]
        this_id = lines[index].split(' ')[2].split('%')[-1]
        for selected_str in selected_strs:
            if this_id == strategy(selected_str).id:
                this_s = selected_str
                break

        if dot_count % LAYER_DOT_NUM == 0:
            this_strs.append('transformer_layer:' +
                             str(dot_count // LAYER_DOT_NUM) + '\n')
        this_strs.append('  ' + dot_name[dot_count % LAYER_DOT_NUM] + ':' +
                         '\n')
        if args.whole_strategy:
            this_strs.append('    ' + 'instruction: ' + this_s)
        this_strs.append('    ' + 'partition: ' +
                         strategy(this_s).selected_str + '\n')

        i = index
        while True:
            i += 1
            if 'Instruction' in lines[i]:
                break
            if args.detailed_partition_strs:
                this_strs.append('    ' + lines[i])
        str_count = i - index - 1
        this_strs.append('    ' + 'total strategy numbers: ' + str(str_count) +
                         '\n')
        strs.append(this_strs)

    return strs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--start_layer',
                        type=int,
                        default=0,
                        help='set the start layer')
    parser.add_argument(
        '--end_layer',
        type=int,
        default=-1,
        help='set the end layer, and generate [start_layer,……,end_layer-1]')
    parser.add_argument('--load_file',
                        type=str,
                        default='log.txt',
                        help='set the loader folder for experiment data')
    parser.add_argument('--save_file',
                        type=str,
                        default='test.txt',
                        help='set the save folder for experiment data')
    parser.add_argument('--whole_strategy',
                        action='store_true',
                        help='show the whole strategy instruction')
    parser.add_argument('--detailed_partition_strs',
                        action='store_true',
                        help='show the partition strategy that can be chosen')
    args = parser.parse_args()

    total_layers = list(range(0, MAX_LAYER + 1))
    layers = total_layers[args.start_layer:args.end_layer]
    f = open(args.load_file, 'r')
    lines = f.readlines()
    indexs = []
    for i in range(len(lines)):
        if 'Startegy Map' in lines[i]:
            start_i = i
        if 'Auto sharding strategy' in lines[i]:
            end_i = i
            break
        end_i = len(lines)
    assert end_i != len(lines)

    for i in range(start_i, end_i):
        if 'dot(' in lines[i]:
            for layer in layers:
                if 'layer/' + str(layer) + '/' in lines[i]:
                    indexs.append(i)
                    break
    selected_strs = []
    start_i = end_i
    for i in range(len(lines)):
        if 'Exit AutoSharding' in lines[i]:
            end_i = i
            break
        end_i = len(lines)
    for i in range(start_i, end_i):
        if 'dot(' in lines[i]:
            selected_strs.append(lines[i])
    selected_strs = selected_strs[1:]
    f.close()
    strs = get_str(args, lines, indexs, selected_strs)
    write_str(args.save_file, strs)
