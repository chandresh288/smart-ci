from sklearn.metrics import accuracy_score
import json

with open("data/evaluation/sentiment_gold.json") as f:
    gold = json.load(f)
with open("data/evaluation/sentiment_predicted.json") as f:
    pred = json.load(f)

y_true = []
y_pred = []

for call_id in gold:
    y_true.append(gold[call_id])
    y_pred.append(pred.get(call_id, "neutral"))  # default fallback

acc = accuracy_score(y_true, y_pred)
print(f"Sentiment Classification Accuracy: {acc:.2f}")
