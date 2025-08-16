#!/usr/bin/env python3

import os
import json
import argparse
import tempfile
from pathlib import Path

import boto3
import pandas as pd
import matplotlib.pyplot as plt

# ─── Helpers ──────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATASET_PATH',      required=True, help='s3://call-recordings-xpia/inbound/dataset.json')
    parser.add_argument('--REPORT_OUTPUT_DIR', required=True, help='s3://call-recordings-xpia/inbound/')
    return parser.parse_args()

def split_s3_path(s3_path):
    assert s3_path.startswith('s3://'), 'Must be an s3:// URI'
    parts = s3_path[5:].split('/', 1)
    return parts[0], parts[1] if len(parts) > 1 else ''

# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    s3 = boto3.client("s3")
    ds_bucket, ds_key = split_s3_path(args.DATASET_PATH)
    out_bucket, out_prefix = split_s3_path(args.REPORT_OUTPUT_DIR)

    with tempfile.TemporaryDirectory() as tmp:
        # 1) Download dataset
        local_ds = os.path.join(tmp, Path(ds_key).name)
        s3.download_file(ds_bucket, ds_key, local_ds)
        with open(local_ds, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df['action_count'] = df['action_items'].apply(len)

        # 2) Sentiment Distribution
        sent_counts = df['sentiment'].str.capitalize().value_counts()
        fig = plt.figure()
        sent_counts.plot(kind='bar')
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Number of Calls')
        plt.tight_layout()
        out_png1 = os.path.join(tmp, "sentiment_distribution.png")
        fig.savefig(out_png1)

        # 3) Action Items Distribution
        fig = plt.figure()
        df['action_count'].plot(
            kind='hist',
            bins=range(df['action_count'].max() + 2),
            rwidth=0.8
        )
        plt.title('Action Items per Call')
        plt.xlabel('Number of Action Items')
        plt.ylabel('Number of Calls')
        plt.tight_layout()
        out_png2 = os.path.join(tmp, "action_items_distribution.png")
        fig.savefig(out_png2)

        # 4) Top 10 Topics
        topics_exploded = df.explode('topics')
        topic_counts = topics_exploded['topics'].value_counts().head(10)
        fig = plt.figure()
        topic_counts.plot(kind='bar')
        plt.title('Top 10 Topics')
        plt.xlabel('Topic')
        plt.ylabel('Frequency')
        plt.tight_layout()
        out_png3 = os.path.join(tmp, "top_topics.png")
        fig.savefig(out_png3)

        # 5) CSV Summaries
        #   a) Sentiment summary
        sentiment_summary = (
            df.groupby('sentiment')['call_id']
              .agg(call_ids=list, num_calls='count')
              .reset_index()
        )
        #   b) Action items summary
        action_summary = (
            df.groupby('action_count')['call_id']
              .agg(call_ids=list, num_calls='count')
              .reset_index()
              .rename(columns={'action_count': 'num_actions'})
        )
        #   c) Topics summary
        topic_summary = (
            topics_exploded.groupby('topics')['call_id']
                           .agg(call_ids=list, frequency='count')
                           .reset_index()
                           .sort_values('frequency', ascending=False)
        )
        #   d) Overall summary
        overall_summary = pd.DataFrame([
            {
                "metric":      "Total Calls",
                "value":       len(df),
                "call_ids":    df['call_id'].tolist()
            },
            {
                "metric":      "Calls with No Topics",
                "value":       int((df['topics'].apply(len) == 0).sum()),
                "call_ids":    df[df['topics'].apply(len) == 0]['call_id'].tolist()
            },
            {
                "metric":      "Calls with No Action Items",
                "value":       int((df['action_count'] == 0).sum()),
                "call_ids":    df[df['action_count'] == 0]['call_id'].tolist()
            },
        ])

        # Write CSVs
        csv_paths = []
        for name, df_sum in [
            ("sentiment_summary.csv", sentiment_summary),
            ("action_items_summary.csv", action_summary),
            ("topic_summary.csv", topic_summary),
            ("overall_summary.csv", overall_summary),
        ]:
            path = os.path.join(tmp, name)
            df_sum.to_csv(path, index=False)
            csv_paths.append(path)

        # 6) Upload everything back to S3
        uploads = [out_png1, out_png2, out_png3] + csv_paths
        for local_file in uploads:
            key = f"{out_prefix}/{Path(local_file).name}"
            s3.upload_file(local_file, out_bucket, key)
            print(f"Uploaded s3://{out_bucket}/{key}")
