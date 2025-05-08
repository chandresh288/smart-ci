#!/usr/bin/env python3

import os
import json
import argparse
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import whisper
from openai import OpenAI

# ─── Helpers ──────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--AUDIO_DIR',      required=True, help='s3://bucket/prefix/')
    parser.add_argument('--OUTPUT_DIR',     required=True, help='s3://bucket/prefix/')
    parser.add_argument('--WHISPER_MODEL',  default='medium')
    parser.add_argument('--MAX_WORKERS',    type=int, default=4)
    parser.add_argument('--OPENAI_API_KEY', required=True)
    return parser.parse_args()

def split_s3_path(s3_path):
    assert s3_path.startswith('s3://'), 'Must be an s3:// URI'
    parts = s3_path[5:].split('/', 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ''
    return bucket, prefix

# Prompt for insight extraction
INSIGHT_PROMPT = """
You are an AI assistant that reads a call transcript and extracts:
1. A concise summary (2–3 sentences).
2. Key topics covered.
3. Action items (who needs to do what by when, if mentioned).
4. Overall sentiment/tone of the call.
Provide output as a JSON with keys: summary, topics, action_items, sentiment.
Transcript:
\"\"\"
{transcript}
\"\"\"
"""

def process_call(key, audio_bucket, out_bucket, out_prefix, whisper_model, openai_client):
    fname = Path(key).stem

    with tempfile.TemporaryDirectory() as tmp:
        local_audio = os.path.join(tmp, Path(key).name)
        # 1) Download audio
        s3.download_file(audio_bucket, key, local_audio)

        # 2) Transcribe
        result = whisper_model.transcribe(local_audio)
        transcript = result["text"]

        # 3) Extract insights
        messages = [
            {"role": "system", "content": "You extract structured call insights as JSON."},
            {"role": "user",   "content": INSIGHT_PROMPT.format(transcript=transcript)}
        ]
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.0,
            max_tokens=500
        )
        content = resp.choices[0].message.content.strip()
        try:
            insights = json.loads(content)
        except json.JSONDecodeError:
            insights = {"error": "Could not parse JSON", "raw": content}

        # 4) Write local files
        txt_file      = os.path.join(tmp, f"{fname}.txt")
        insights_file = os.path.join(tmp, f"{fname}_insights.json")
        with open(txt_file,      "w", encoding="utf-8") as f: f.write(transcript)
        with open(insights_file, "w", encoding="utf-8") as f: json.dump(insights, f, ensure_ascii=False, indent=2)

        # 5) Upload to S3
        base = f"{out_prefix}/{fname}"
        s3.upload_file(txt_file,      out_bucket, f"{base}/{fname}.txt")
        s3.upload_file(insights_file, out_bucket, f"{base}/{fname}_insights.json")

        print(f"[Done] {fname}")

# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    # S3 clients & paths
    s3 = boto3.client("s3")
    audio_bucket, audio_prefix = split_s3_path(args.AUDIO_DIR)
    out_bucket,   out_prefix   = split_s3_path(args.OUTPUT_DIR)

    # Models & clients
    whisper_model   = whisper.load_model(args.WHISPER_MODEL)
    openai_client   = OpenAI(api_key=args.OPENAI_API_KEY)

    # List audio files
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=audio_bucket, Prefix=audio_prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.lower().endswith((".mp3", ".wav", ".m4a", ".flac")):
                keys.append(k)

    print(f"Found {len(keys)} audio files in s3://{audio_bucket}/{audio_prefix}")

    # Parallel processing
    with ThreadPoolExecutor(max_workers=args.MAX_WORKERS) as exe:
        futures = [
            exe.submit(
                process_call, k,
                audio_bucket, out_bucket, out_prefix,
                whisper_model, openai_client
            ) for k in keys
        ]
        for fut in as_completed(futures):
            fut.result()
