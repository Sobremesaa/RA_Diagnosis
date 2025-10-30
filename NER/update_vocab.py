from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer("../../../bert-base-chinese/vocab.txt")
text = 'jjj今天天气很好。'
tokens = tokenizer.tokenize(text)
print('未添加新词前:', tokens)
tokenizer.add_tokens('jjj')
tokens = tokenizer.tokenize(text)
print('添加新词后:', tokens)
