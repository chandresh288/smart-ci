import os
import json
from faker import Faker
import random

# Setup
base_dir = "data/evaluation"
os.makedirs(f"{base_dir}/manual_transcripts", exist_ok=True)
os.makedirs(f"{base_dir}/whisper_transcripts", exist_ok=True)

fake = Faker()
topics = ["billing", "technical issue", "late fee", "installation", "connectivity", "router", "account", "password reset", "fraud alert", "plan upgrade"]
sentiments = ["positive", "neutral", "negative", "frustrated", "inquisitive", "confused"]
actions = ["call back", "send email", "escalate issue", "schedule technician", "verify identity", "update records", "reset password", "issue refund"]

topic_gold = {}
topic_pred = {}
sentiment_gold = {}
sentiment_pred = {}
action_gold = {}
action_pred = {}

# Generate 20 sample files and annotations
for i in range(1, 120):
    call_id = f"call_{i:03d}"
    text = fake.paragraph(nb_sentences=5)

    # Save manual and Whisper transcriptions (slight variations)
    with open(f"{base_dir}/manual_transcripts/{call_id}.txt", "w") as f:
        f.write(text)
    with open(f"{base_dir}/whisper_transcripts/{call_id}.txt", "w") as f:
        f.write(text.replace("the", "").replace("a", "an"))  # simple simulated WER variation

    # Annotations
    true_topics = random.sample(topics, k=random.randint(1, 3))
    pred_topics = random.sample(topics, k=random.randint(1, 3))
    topic_gold[call_id] = true_topics
    topic_pred[call_id] = pred_topics

    sentiment_gold[call_id] = random.choice(sentiments)
    sentiment_pred[call_id] = random.choice(sentiments)

    true_actions = random.sample(actions, k=random.randint(1, 2))
    pred_actions = random.sample(actions, k=random.randint(1, 2))
    action_gold[call_id] = true_actions
    action_pred[call_id] = pred_actions

# Save JSON files
with open(f"{base_dir}/topic_gold.json", "w") as f:
    json.dump(topic_gold, f, indent=2)
with open(f"{base_dir}/topic_predicted.json", "w") as f:
    json.dump(topic_pred, f, indent=2)
with open(f"{base_dir}/sentiment_gold.json", "w") as f:
    json.dump(sentiment_gold, f, indent=2)
with open(f"{base_dir}/sentiment_predicted.json", "w") as f:
    json.dump(sentiment_pred, f, indent=2)
with open(f"{base_dir}/action_gold.json", "w") as f:
    json.dump(action_gold, f, indent=2)
with open(f"{base_dir}/action_predicted.json", "w") as f:
    json.dump(action_pred, f, indent=2)
