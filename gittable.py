import os
import random
from pyarrow import parquet as pq
import json


# organism subset
path = 'data/organism_tables'
files = os.listdir(path)
data = []
num_rows = 0
num_columns = 0
random.seed(2021)
bad_title = 0

for file in files:
    try:
        table = pq.read_table(os.path.join(path, file))
    except:
        print('Error: {}'.format(file))
    else:
        d = {}
        all_headers = table.column_names
        header2id = {}
        nan_set = set()
        for index, h in enumerate(all_headers):
            header2id[h] = index
        all_columns = table.columns
        for i in range(len(all_columns)):
            all_columns[i] = all_columns[i].to_pandas()
            if all_columns[i].isna().any():
                nan_set.add(i)
            all_columns[i] = all_columns[i].tolist()
            temp = set(all_columns[i])
            if len(temp) <= 1:
                nan_set.add(i)

        metadata = table.schema.metadata
        metadata = metadata[b'gittables']
        metadata = str(metadata, 'utf-8')
        metadata = json.loads(metadata)
        title = metadata['table_domain']['schema_embedding']
        # remove bad title
        if title is None:
            continue
        headers = []
        columns = []
        labels = []
        flag = False
        t_columns = 0
        for key, value in metadata['schema_embedding_column_types'].items():
            c_id = header2id[key]
            if c_id in nan_set:
                continue
            headers.append(all_headers[c_id])
            columns.append(all_columns[c_id])
            labels.append(value['cleaned_label'])
            num_columns += 1
            t_columns += 1
            flag = True
        if flag:
            d['title'] = title
            d['headers'] = headers
            d['columns'] = columns
            t_rows = 0
            for item in columns:
                num_rows += len(item)
                t_rows += len(item)
            if t_rows == 0 or t_columns == 0:
                continue
            d['labels'] = labels
            d['num_rows'] = t_rows
            d['num_columns'] = t_columns
            data.append(d)

# train/valid/test 8:1:1
tables = {}
random.shuffle(data)
num_tables = len(data)
k = num_tables // 10
tables['train'] = data[0: 8 * k]
tables['dev'] = data[8 * k: 9 * k]
tables['test'] = data[9 * k: num_tables]

labels_set = set()
for item in tables['train']:
    temp = item['labels']
    for label in temp:
        labels_set.add(label)

for item in tables['dev']:
    temp = item['labels']
    for label in temp:
        if label not in labels_set:
            num_rows -= item['num_rows']
            num_columns -= item['num_columns']
            tables['dev'].remove(item)
            break
for item in tables['test']:
    temp = item['labels']
    for label in temp:
        if label not in labels_set:
            num_rows -= item['num_rows']
            num_columns -= item['num_columns']
            tables['test'].remove(item)
            break

for file in ['train', 'dev', 'test']:
    output_path = './data/GitTable/{}.table_col_type.json'.format(file)
    with open(output_path, 'w') as write_file:
        json.dump(tables[file], write_file)

output_path = './data/GitTable/type_vocab.txt'
with open(output_path, 'w') as write_file:
    for index, item in enumerate(labels_set):
        temp = str(index) + '\t' + str(item) + '\n'
        write_file.write(temp)

num_tables = len(tables['train']) + len(tables['dev']) + len(tables['test'])
print('Num train: {}  Num valid: {}  Num test: {}'.format(len(tables['train']), len(tables['dev']), len(tables['test'])))
print('Num tables: {}'.format(num_tables))
print('Num rows: {}'.format(num_rows))
print('Num columns: {}'.format(num_columns))
