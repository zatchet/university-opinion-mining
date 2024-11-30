import torch
from transformers import BertTokenizer
from bert_classifier import BERTClassifier

model = BERTClassifier(bert_model_name='bert-base-uncased', num_classes=2)
model.load_state_dict(torch.load("fact_opinion_classification/bert_classifier.pth", map_location=torch.device('cpu')))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(text):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
    return "fact" if preds.item() == 1 else "opinion"

print(predict('i love zebras'))
print(predict('zach is a student'))