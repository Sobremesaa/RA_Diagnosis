from bert_model import MyModel
from config import parsers
import torch
from transformers import BertTokenizer
import time
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def load_model(device, model_path):
    myModel = MyModel().to(device)
    myModel.load_state_dict(torch.load(model_path, weights_only=False))
    myModel.eval()
    return myModel


def process_text(text, bert_pred):
    tokenizer = BertTokenizer.from_pretrained(bert_pred)

    # 读取新词文件并逐行添加到tokenizer
    with open(parsers().new_vocab, 'r', encoding='utf-8') as vocab_file:
        new_tokens = vocab_file.read().splitlines()
    tokenizer.add_tokens(new_tokens)

    token_id = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokenizer.tokenize(text))
    mask = [1] * len(token_id) + [0] * (args.max_len + 2 - len(token_id))
    token_ids = token_id + [0] * (args.max_len + 2 - len(token_id))
    token_ids = torch.tensor(token_ids).unsqueeze(0)
    mask = torch.tensor(mask).unsqueeze(0)
    x = torch.stack([token_ids, mask])
    return x

def text_class_name(pred):
    # 应用softmax得到概率分布
    probabilities = F.softmax(pred, dim=1)
    # 获取最大概率及其索引
    max_prob, predicted_index = torch.max(probabilities, dim=1)
    result_index = predicted_index.cpu().numpy().tolist()
    result_prob = max_prob.cpu().numpy().tolist()

    classification = open(args.classification, "r", encoding="utf-8").read().split("\n")
    classification_dict = dict(zip(range(len(classification)), classification))

    result_category = result_index[0]
    result_prob = round(result_prob[0], 4)
    print("result_category---", result_category)
    # 打印文本、预测类别及对应的置信度
    print(f"文本：{text}\n预测的类别为：{classification_dict[result_index[0]]}\t置信度：{result_prob:.4f}")
    return [result_category,result_prob]

# def text_class_name(pred):
#     result = torch.argmax(pred, dim=1)
#     result = result.cpu().numpy().tolist()
#     classification = open(args.classification, "r", encoding="utf-8").read().split("\n")
#     classification_dict = dict(zip(range(len(classification)), classification))
#     print(f"文本：{text}\n预测的类别为：{classification_dict[result[0]]}")
    
    
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    splits = np.load('D:/master/code/data/data_splits.npz')
    csv_file = "D:/master/code/data/data_all_enhanced_1.csv"
    data = pd.read_csv(csv_file, encoding='utf-8')
    data['label'] = data['label'].apply(lambda x: 1 if x == 1 else 0)

    print(data.head())

    test_data = data.iloc[splits['test']]

    X = test_data.iloc[:, 7:8]
    X = X.to_numpy().ravel()

    y = test_data.iloc[:, 6:7]
    y = y.to_numpy().ravel()
    print("y---", y)




    start = time.time()
    args = parsers()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    #
    model = load_model(device, args.save_model_best)
    #
    # data = pd.read_csv('D://master//work-code//dataProcess//test_3//test_data.csv')
    # X = test_data.iloc[:, 7:8]
    # X = X.to_numpy().ravel()

    # texts = ["我们一起去打篮球吧！", "我喜欢踢足球！", "沈腾和马丽的新电影《独行月球》很好看", "昨天玩游戏，完了一整天", "现在的高考都已经开始分科考试了。", "中方：佩洛西如赴台将致严重后果",
    # "现在的股票基金趋势很不好"] texts = ["双手近端指间关节、掌指关节、双腕关节肿胀、疼痛，晨僵时间大于1小时，双肘、双肩、双膝及双足跖趾关节疼痛，无肿胀","左足第1跖趾关节外侧压痛，轻度肿胀，局部轻度发红。",
    # "双肩压痛、活动痛，双上肢上抬困难，左肘关节肿胀、压痛，右肘关节压痛，屈曲伸直尚可，双腕关节肿胀、压痛，背伸受限，双手第2、3近端指间、掌指关节肿胀、压痛，转颈受限，胸腰椎后凸畸形，脊柱生理弯曲消失。双侧4
    # 字试验不能配合，骶髂部无叩痛，双髋关节内收外展活动轻度受限。双膝关节肿胀、压痛，屈曲受限，双踝关节肿胀、压痛，双足背肿胀，双足底压痛。枕墙距8cm，胸廓活动度2cm，指地距30cm，Schober试验2cm。",
    # "双肩关节压痛、活动痛，双上肢上抬受限，双肘关节压痛，无肿胀，关节活动可，双腕关节肿胀、压痛，背伸受限，双手各指间关节无肿胀、压痛，双膝关节压痛，无肿胀，关节活动可，下蹲后起身困难，双踝关节肿胀、压痛，双足底压痛。",
    # "双肩关节肿胀压痛，双腕关节肿胀压痛，双膝关节稍压痛，双手近端之间关节肿胀压痛"]
    texts = X
    print("texts---", texts)
    print("模型预测结果：")
    result = [[],[]]
    cls_vectors = []  # 用于存储CLS向量
    for text in texts:
        x = process_text(text, args.bert_pred)
        # print("x---",x)
        with torch.no_grad():
            pred, cls_vector = model(x)
        # logits = pred.logits
        # print("pred---", pred)
        # print("cls_vectors---", cls_vector.shape)
        cls_vectors.append(cls_vector.cpu().detach().numpy())
        temp = text_class_name(pred)
        result[0].append(temp[0])
        result[1].append(temp[1])
        # multi_hot_code.append(temp[1].item())
    end = time.time()

    # ------------------- CSV输出核心代码 -------------------
    # 1. 构建DataFrame（两列：预测概率、真实标签）
    result_df = pd.DataFrame({
        "un_pro": result[0],
        "un_res": result[1],
        "label": test_data["label"]
    })

    # 2. 保存为CSV文件（可修改文件路径，如"./model_predictions.csv"）
    output_path = "D://master/code/result/un_result.csv"  # 输出文件路径
    result_df.to_csv(output_path, index=False, encoding="utf-8")  # index=False不保存行号

    print("降维前---",  len(cls_vectors), cls_vectors[0].shape)
    # print("降维前---", cls_vectors)
    # 将多个样本的 CLS 向量合并成一个二维数组
    cls_vectors_np = np.vstack(cls_vectors)
    # 创建 PCA 对象并指定降维后的维度为 32
    pca = PCA(n_components=32)
    # 进行降维操作
    downsampled_cls_vectors = pca.fit_transform(cls_vectors_np)
    print("降维后---", downsampled_cls_vectors.shape)
    # print("降维后---", downsampled_cls_vectors)


    # print("预测结果---", pred)
    # print("CLS向量---", cls_vectors)

    # print("result---", result)

    # save_path = "D://master//work-code//dataProcess//test_3//"
    # result_0 = np.array(result[0])
    # df0 = pd.DataFrame(result_0)
    # df0.to_csv(save_path+'test_result0.csv', index=False, header=False)
    # result_1 = np.array(result[1])
    # df1 = pd.DataFrame(result_1)
    # df1.to_csv(save_path+'unstru_result.csv', index=False, header=False)

    # # 独热编码
    # labels = np.array(result[0])
    # # 创建一个独热编码器对象
    # encoder = OneHotEncoder(categories='auto')
    # # 将数组reshape成一列
    # labels = labels.reshape(-1, 1)
    # # 进行独热编码
    # one_hot_encoded = encoder.fit_transform(labels).toarray()
    # print("标签独热编码---", one_hot_encoded)
    # df2 = pd.DataFrame(one_hot_encoded)
    # df2.to_csv('result_one_hot_0.csv', index=False, header=False)
    #
    # temp = result_1
    # one_hot_encoded_1 = one_hot_encoded * result_1[:, np.newaxis]
    # df3 = pd.DataFrame(one_hot_encoded_1)
    # df3.to_csv('result_one_hot_1.csv', index=False, header=False)

    print(f"耗时为：{end - start} s")
