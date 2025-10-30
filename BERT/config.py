import argparse
import os.path

# data_path = "./data"
vocab_path = "./data/vocab"
data_path = "./data/after_enhance"
model_path = "./model/after_enhance"

def parsers():
    parser = argparse.ArgumentParser(description="Bert model of argparse")
    parser.add_argument("--train_file", type=str, default=os.path.join(data_path, "train_dev.txt"))
    parser.add_argument("--dev_file", type=str, default=os.path.join(data_path, "dev.txt"))
    parser.add_argument("--test_file", type=str, default=os.path.join(data_path, "test.txt"))
    parser.add_argument("--classification", type=str, default=os.path.join(data_path, "class.txt"))
    parser.add_argument("--new_vocab", type=str, default=os.path.join(vocab_path, "new_vocab.txt"))
    parser.add_argument("--bert_pred", type=str, default="D:/app/bert-model/bert-base-chinese")
    parser.add_argument("--split_data_dir", type=str, default="./model/2_class_enhance/split")
    parser.add_argument("--class_num", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=230)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learn_rate", type=float, default=1e-5)
    parser.add_argument("--num_filters", type=int, default=768)
    parser.add_argument("--save_model_best", type=str, default=os.path.join(model_path, "RA_bert_model.pth"))
    parser.add_argument("--save_model_last", type=str, default=os.path.join(model_path, "RA_last_model.pth"))
    args = parser.parse_args()
    return args
