# text = '左腕关节轻度肿胀、压痛、活动痛，左腕背伸及掌屈功能受限。右肩关节上举轻度受限'
# chunks_extract = [{'st_idx': 0, 'end_idx': 1, 'label': 'Position'}, {'st_idx': 1, 'end_idx': 4, 'label': 'Body'}, {'st_idx': 4, 'end_idx': 6, 'label': 'Degree'}, {'st_idx': 6, 'end_idx': 8, 'label': 'Signs'}, {'st_idx': 9, 'end_idx': 11, 'label': 'Signs'}, {'st_idx': 12, 'end_idx': 15, 'label': 'Signs'}, {'st_idx': 16, 'end_idx': 17, 'label': 'Position'}, {'st_idx': 17, 'end_idx': 18, 'label': 'Body'}, {'st_idx': 18, 'end_idx': 20, 'label': 'Function'}, {'st_idx': 28, 'end_idx': 29, 'label': 'Position'}, {'st_idx': 29, 'end_idx': 32, 'label': 'Body'}, {'st_idx': 32, 'end_idx': 34, 'label': 'Function'}, {'st_idx': 34, 'end_idx': 36, 'label': 'Degree'}, {'st_idx': 36, 'end_idx': 38, 'label': 'Signs'}]
#
# # 根据chunks_extract中的信息提取有意义的部分并进行拼接
# extracted_parts = []
# for chunk in chunks_extract:
#     start_idx = chunk['st_idx']
#     end_idx = chunk['end_idx']
#     extracted_part = text[start_idx:end_idx]
#     extracted_parts.append(extracted_part)
#
# # 将提取出的有意义部分进行拼接
# result = " ".join(extracted_parts)
#
# # 输出拼接后的结果
# print(result)

import torch
import pandas as pd
import pickle
import csv

from BiLSTM_CRF.data_processor import get_vocab, get_label_map
from BiLSTM_CRF.model import BiLSTM_CRF

def chunks_extract(pred):
    if not pred:
        return []

    cur_entity = None
    res = []
    st_idx, end_idx = 0, 0
    for i, pred_single in enumerate(pred):
        pred_start_B = pred_single.startswith('B')
        pred_entity = pred_single.split('-')[-1]

        if cur_entity:
            if pred_start_B or cur_entity != pred_entity:
                res.append({
                    'st_idx': st_idx,
                    'end_idx': i,
                    'label': cur_entity
                })
                cur_entity = None
        if pred_start_B:
            st_idx = i
            cur_entity = pred_entity
    if cur_entity:
        res.append({
            'st_idx': st_idx,
            'end_idx': len(pred),
            'label': cur_entity,
        })
    return res


def predict(text):
    embedding_size = 128
    hidden_dim = 768

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # 加载训练时保存的词表
    vocab = get_vocab()[0]
    # 加载训练时保存的标签字典
    label_map, label_map_inv = get_label_map()
    model = BiLSTM_CRF(embedding_size, hidden_dim, vocab, label_map, device)
    model.load_state_dict(torch.load('model_91.bin'))
    model.to(device)

    model.eval()
    model.state = 'pred'
    with torch.no_grad():
        text = [vocab.get(t, vocab['UNK']) for t in text]
        seq_len = torch.tensor(len(text), dtype=torch.long).unsqueeze(0)
        seq_len = seq_len.to(device)
        text = torch.tensor(text, dtype=torch.long).unsqueeze(0)
        text = text.to(device)
        batch_tag = model(text, seq_len)
        pred = [label_map_inv[t] for t in batch_tag]

    return pred

# text = '左腕关节轻度肿胀、压痛、活动痛，左腕背伸及掌屈功能受限。右肩关节上举轻度受限'
# chunks_extract = [{'st_idx': 0, 'end_idx': 1, 'label': 'Position'}, {'st_idx': 1, 'end_idx': 4, 'label': 'Body'}, {'st_idx': 4, 'end_idx': 6, 'label': 'Degree'}, {'st_idx': 6, 'end_idx': 8, 'label': 'Signs'}, {'st_idx': 9, 'end_idx': 11, 'label': 'Signs'}, {'st_idx': 12, 'end_idx': 15, 'label': 'Signs'}, {'st_idx': 16, 'end_idx': 17, 'label': 'Position'}, {'st_idx': 17, 'end_idx': 18, 'label': 'Body'}, {'st_idx': 18, 'end_idx': 20, 'label': 'Function'}, {'st_idx': 28, 'end_idx': 29, 'label': 'Position'}, {'st_idx': 29, 'end_idx': 32, 'label': 'Body'}, {'st_idx': 32, 'end_idx': 34, 'label': 'Function'}, {'st_idx': 34, 'end_idx': 36, 'label': 'Degree'}, {'st_idx': 36, 'end_idx': 38, 'label': 'Signs'}]

def entity_joint(text, chunks_extract):
    # 根据chunks_extract中的信息提取有意义的部分并按照label排序
    extracted_parts = []
    for chunk in sorted(chunks_extract, key=lambda x: x['st_idx']):
        start_idx = chunk['st_idx']
        end_idx = chunk['end_idx']
        extracted_part = text[start_idx:end_idx]
        extracted_parts.append((chunk['label'], extracted_part))

    # 根据label顺序拼接成有意义的句子
    result = ""
    for label, part in extracted_parts:
        if label in ['Position', 'Body', 'Function', 'Degree', 'Signs']:
            result += part + " "

    return result


def creat_vocab(joint_data):
    # 词表保存路径
    vocab_path = '../mydata/vocab_1.pkl'

    line_vacab = []
    for line in joint_data:
        print('line-----', line)
        rows = line.split(' ')
        print("rows---",rows)
        for vocab in rows:
        # if len(rows) != 1:
            line_vacab.append(vocab)
    # 建立词表字典，提前加入'PAD'和'UNK'
    # 'PAD'：在一个batch中不同长度的序列用该字符补齐
    # 'UNK'：当验证集或测试集出现词表以外的词时，用该字符代替
    vocab = {'PAD': 0, 'UNK': 1}
    # 遍历数据集，不重复取出所有字符，并记录索引
    for data in line_vacab:  # 获取实体标签，如'name'，'compan
        # print(data)
        if data not in vocab:
            vocab[data] = len(vocab)
    # vocab：{'PAD': 0, 'UNK': 1, '浙': 2, '商': 3, '银': 4, '行': 5...}
    # 保存成csv文件
    with open('../mydata/vocab_1.csv','w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in vocab:
            row = [row]
            # print("row---",row)
            writer.writerow(row)

    # 保存成pkl文件
    with open(vocab_path, 'wb') as fp:
        pickle.dump(vocab, fp)
    print('vocab-------')
    print(vocab)

    # 翻转字表，预测时输出的序列为索引，方便转换成中文汉字
    # vocab_inv：{0: 'PAD', 1: 'UNK', 2: '浙', 3: '商', 4: '银', 5: '行'...}

    vocab_inv = {v: k for k, v in vocab.items()}
    return vocab, vocab_inv


# 读取CSV文件
df = pd.read_csv('../mydata/MultiRAText.csv')

joint_data = []
for text in df['text']:
    print("text---",text)
    # text = '左腕关节轻度肿胀、压痛、活动痛，左腕背伸及掌屈功能受限。右肩关节上举轻度受限'
    pred = predict(text)
    print("pred---", pred)

    chunks = chunks_extract(pred)
    # print("chunks-----",chunks)

    joint_result = entity_joint(text, chunks)
    joint_data.append(joint_result)
    print("joint_result---",joint_result)

print("joint_data---",joint_data)
creat_vocab(joint_data)
