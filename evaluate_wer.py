from jiwer import wer
import json
import os

# Paths to true and predicted transcripts
true_dir = "data/evaluation/manual_transcripts"
pred_dir = "data/evaluation/whisper_transcripts"

wer_scores = []

for fname in os.listdir(true_dir):
    if fname.endswith(".txt"):
        with open(os.path.join(true_dir, fname)) as f:
            reference = f.read().strip()
        with open(os.path.join(pred_dir, fname)) as f:
            hypothesis = f.read().strip()
        wer_score = wer(reference, hypothesis)
        wer_scores.append(wer_score)

print("Average WER:", round(sum(wer_scores) / len(wer_scores), 4))
