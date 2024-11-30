import torch
from transformers import BertTokenizer
from bert_classifier import BERTClassifier

model = BERTClassifier(bert_model_name='bert-base-uncased', num_classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("fact_opinion_classification/model.pth", map_location=device))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model.to(device)

def predict(text):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
    return "fact" if preds.item() == 1 else "opinion"

print(predict('i think zebras are awesome'))
print(predict('zach is a student'))