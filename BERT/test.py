import torch
from tqdm import tqdm
from utils import read_data, MyDataset
from config import parsers
from torch.utils.data import DataLoader
from bert_model import MyModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report


def test_data():
    args = parsers()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    test_text, test_label = read_data(args.test_file)
    test_dataset = MyDataset(test_text, test_label, args.max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = MyModel().to(device)
    model.load_state_dict(torch.load(args.save_model_best, weights_only=False))
    model.eval()

    all_pred, all_true = [], []
    with torch.no_grad():
        for batch_text, batch_label in tqdm(test_dataloader):
            batch_label, batch_label = batch_label.to(device), batch_label.to(device)
            pred = model(batch_text)
            pred = torch.argmax(pred, dim=1)

            pred = pred.cpu().numpy().tolist()
            label = batch_label.cpu().numpy().tolist()

            print("batch_text---", batch_text)
            print("pred---", pred, "---label---", label)

            all_pred.extend(pred)
            all_true.extend(label)

    accuracy = accuracy_score(all_true, all_pred)
    precision = precision_score(all_true, all_pred, pos_label = 0)
    # recall = recall_score(all_true, all_pred, average='weighted', zero_division=0)
    recall = recall_score(all_true, all_pred, pos_label = 0)
    f1 = f1_score(all_true, all_pred, pos_label = 0)
    auc = roc_auc_score(all_true, all_pred, multi_class='ovr', average='weighted')

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_true, all_pred)

    # 提取TP, TN, FP, FN
    # 这里以二分类为例，如果是多分类需要另外处理
    TP = conf_matrix[1, 1]  # True Positives
    TN = conf_matrix[0, 0]  # True Negatives
    FP = conf_matrix[0, 1]  # False Positives
    FN = conf_matrix[1, 0]  # False Negatives

    print(f"---------------test dataset result---------------")
    print(f"accuracy:{accuracy:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1-score:{f1:.4f}, AUC-ROC{auc:.4f} ")

    print("真实正例 (TP):", TP)
    print("真实负例 (TN):", TN)
    print("假正例 (FP):", FP)
    print("假负例 (FN):", FN)
    print(classification_report(all_true, all_pred))

if __name__ == "__main__":
    test_data()
