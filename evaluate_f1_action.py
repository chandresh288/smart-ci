from sklearn.metrics import precision_recall_fscore_support
import json

with open("data/evaluation/action_gold.json") as f:
    gold = json.load(f)
with open("data/evaluation/action_predicted.json") as f:
    pred = json.load(f)

all_true = []
all_pred = []

for call_id in gold:
    all_true.append(set(gold[call_id]))  # true topics
    all_pred.append(set(pred.get(call_id, [])))  # predicted topics

# Flatten sets
true_labels = []
pred_labels = []

for i in range(len(all_true)):
    for action in gold[call_id] + pred.get(call_id, []):
        true_labels.append(int(action in all_true[i]))
        pred_labels.append(int(action in all_pred[i]))

p, r, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary')
print(f"Action Extraction - F1 Score: {f1:.2f}")
