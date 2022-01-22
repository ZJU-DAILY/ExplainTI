import os
import json
import argparse
from tqdm import tqdm
import numpy as np


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=int, default=16)
    parser.add_argument('--data_name', type=str, default='GitTable')
    args = parser.parse_args(args)
    return args


if __name__ == '__main__':
    args = parse_args()
    modes = ['col_type', 'rel_extraction']
    sample_size = args.sample_size
    data_name = args.data_name
    seed = 2021
    np.random.seed(seed)

    print('Sample size: {}'.format(sample_size))
    path = os.path.join('data/', data_name)
    for mode in modes:
        if data_name == 'GitTable' and mode == 'rel_extraction':
            break
        print('Dataset: {}  mode: {}'.format(data_name, mode))
        title2column = {}
        header2column = {}
        titles = set()
        headers = set()
        column2id = {}
        pair2id = {}

        index = 0
        for prefix in ['train', 'dev', 'test']:
            file = '{}.table_{}.json'.format(prefix, mode)
            table_path = os.path.join(path, file)
            with open(table_path) as table_file:
                table = json.load(table_file)
                for t in tqdm(table):
                    if data_name == 'WikiTable':
                        title = t[1].strip()
                    else:
                        title = t['title'].strip()
                    titles.add(title)
                    if mode == 'col_type':
                        if data_name == 'GitTable':
                            t[6] = t['columns']
                        for i, column in enumerate(t[6]):
                            if data_name == 'WikiTable':
                                header = t[5][i].strip()
                                column_name = t[0] + '-' + header
                            else:
                                header = t['headers'][i].strip()
                                column_name = title + '-' + header
                            headers.add(header)
                            column2id[column_name] = index
                            index += 1
                            if title not in title2column:
                                title2column[title] = []
                            if header not in header2column:
                                header2column[header] = []
                            title2column[title].append(column2id[column_name])
                            header2column[header].append(column2id[column_name])
                    else:
                        for i, column in enumerate(t[6]):
                            header = t[5][i].strip()
                            if i == 0:
                                left_column_name = t[0] + '-' + header
                                left_header = header
                            else:
                                right_column_name = t[0] + '-' + header
                                pair_name = left_column_name + right_column_name
                                pair_header = left_header + header
                                pair2id[pair_name] = index
                                index += 1
                                if title not in title2column:
                                    title2column[title] = []
                                if pair_header not in header2column:
                                    header2column[pair_header] = []
                                title2column[title].append(pair2id[pair_name])
                                header2column[pair_header].append(pair2id[pair_name])

        if mode == 'col_type':
            print('Num columns: {}'.format(index))
        else:
            print('Num pairs: {}'.format(index))

        for prefix in ['train', 'dev', 'test']:
            file = '{}.table_{}.json'.format(prefix, mode)
            output_file = '{}.graph_{}.npy'.format(prefix, mode)
            output_file = os.path.join(path, output_file)
            print('Build Column Graph {}'.format(file))
            column_graph = []
            table_path = os.path.join(path, file)
            with open(table_path) as table_file:
                table = json.load(table_file)
                for t in tqdm(table):
                    if data_name == 'WikiTable':
                        title = t[1].strip()
                    else:
                        title = t['title'].strip()
                    if mode == 'col_type':
                        if data_name == 'GitTable':
                            t[6] = t['columns']
                        for i, column in enumerate(t[6]):
                            if data_name == 'WikiTable':
                                header = t[5][i].strip()
                            else:
                                header = t['headers'][i].strip()
                            neighbors = header2column[header] + title2column[title]
                            neighbors = np.array(neighbors)
                            neighbors = np.random.choice(neighbors, size=sample_size)
                            column_graph.append(neighbors)
                    else:
                        for i, column in enumerate(t[6]):
                            header = t[5][i].strip()
                            if i == 0:
                                left_header = header
                            else:
                                pair_header = left_header + header
                                neighbors = header2column[pair_header] + title2column[title]
                                neighbors = np.array(neighbors)
                                neighbors = np.random.choice(neighbors, size=sample_size)
                                column_graph.append(neighbors)
            column_graph = np.array(column_graph)
            np.save(output_file, column_graph)
