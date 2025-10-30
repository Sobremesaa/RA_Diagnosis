import torch
import pandas as pd
import numpy as np
from BiLSTM_CRF.data_processor import get_vocab, get_label_map
from BiLSTM_CRF.model import BiLSTM_CRF

embedding_size = 128
hidden_dim = 768

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


def predict(origin_text):
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
        text = [vocab.get(t, vocab['UNK']) for t in origin_text]
        seq_len = torch.tensor(len(text), dtype=torch.long).unsqueeze(0)
        seq_len = seq_len.to(device)
        text = torch.tensor(text, dtype=torch.long).unsqueeze(0)
        text = text.to(device)
        batch_tag = model(text, seq_len)
        pred = [label_map_inv[t] for t in batch_tag]

    return pred
        # chunks_extract = chunks_extract(pred)
        # print(len(pred))
        # print("pred---",type(pred),"---",pred)
        # print("chunks_extract---",type(chunks_extract),"---",chunks_extract)


def rejoin_entities(text, chunks_extract):
    # print("text-----",text)
    # print("chunks_extract-----",chunks_extract)
    entities = []
    for chunk in chunks_extract:
        entity_text = text[chunk['st_idx']:chunk['end_idx']]
        entities.append((entity_text, chunk['label']))

    new_entities = []
    current_entity = None

    for i, (token, tag) in enumerate(zip(text, pred)):
        if tag.startswith('B-'):
            if current_entity:
                new_entities.append(current_entity)
            current_entity = {'text': token, 'label': tag.split('-')[1]}
        elif tag.startswith('I-'):
            if current_entity:
                current_entity['text'] += token
        elif tag == 'O':
            if current_entity:
                new_entities.append(current_entity)
                current_entity = None

    if current_entity:
        new_entities.append(current_entity)

    reconstructed_entities = []
    for entity in new_entities:
        for original_entity in entities:
            if entity['text'] == original_entity[0] and entity['label'] == original_entity[1]:
                reconstructed_entities.append((entity['text'], entity['label']))
                break

    return reconstructed_entities

# print("实体---",rejoined_entities)

# def combine_entities_without_text(rejoined_entities):
#     sentence = ''
#     # current_idx = 0
#
#     for entity_text, _ in rejoined_entities:
#         # st_idx = current_idx
#         # end_idx = st_idx + len(entity_text)
#         sentence += entity_text
#         # current_idx = end_idx
#
#     return sentence
#
# combined_sentence = combine_entities_without_text(rejoined_entities)
# print(combined_sentence)
def combine_entities(rejoined_entities):
    index = 0
    flag = False # 标记是否有未拼接的
    sentences = []
    sentence = ""
    start = 0
    end = 0
    for entity,label in rejoined_entities:
        if label == "Position":
            start = 1
            if index != 0: sentences.append(sentence)
            sentence = ""
            sentence += entity
        elif label == "Body":
            if start == 1:
                sentence += entity
            else:
                start = 1
                if index != 0: sentences.append(sentence)
                sentence = ""
                sentence += entity
        elif label == "Number":
            print(entity)
            entity = entity.replace("、","")
            entity = entity.replace("各", "第2345")
            trantab = str.maketrans("一二三四五","12345")
            entity = entity.translate(trantab)
            text_list = entity.split("-")
            print("text_list-----", text_list)
            if len(text_list)>1:
                i = 0
                num_sentence = ""
                for num_text in text_list:
                    if i <= len(text_list) - 2:
                        temp_1 = num_text[len(num_text)-1]
                        temp_2 = text_list[i+1][0]
                        print("temp-----", temp_1, temp_2)
                        for j in range(int(temp_1), int(temp_2)+1):
                            num_sentence += str(j);
                    i += 1
                sentence += num_sentence
            else: sentence += text_list[0]
        elif label == "Degree":
            sentence += entity
        elif label == "Signs":
            if rejoined_entities[index-1][1] == "Signs":
                sentence = sentence + '、' + entity
            else: sentence += entity
        elif label == "Function":
            if rejoined_entities[index - 1][1] == "Function":
                sentence = sentence + '、' + entity
            else: sentence += entity
        index += 1
    sentences.append(sentence)
    return sentences

# 读取CSV文件
df = pd.read_csv('../mydata/MultiRAText.csv')
# print("df-----",df['text'])

for text in df['text']:
    print("text---",text)
    # text = '左腕关节轻度肿胀、压痛、活动痛，左腕背伸及掌屈功能受限。右肩关节上举轻度受限'
    pred = predict(text)
    # print("pred-----",pred)
    chunks = chunks_extract(pred)
    print("chunks-----",chunks)

    rejoined_entities = rejoin_entities(text, chunks)

    combined_sentence = combine_entities(rejoined_entities)
    print("combined_sentence---",combined_sentence)


