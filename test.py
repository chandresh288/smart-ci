import os
import json
import glob
import whisper
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ─── Configuration ─────────────────────────────────────────────────────────────
# Directory containing your audio files
AUDIO_DIR    = "calls_audio_2/"
# Where to write transcripts and insights
OUTPUT_DIR   = "calls_output/"
# Whisper model size: tiny, base, small, medium, large
WHISPER_MODEL = "medium"
# Number of threads to parallelize on
MAX_WORKERS  = 4

# Instantiate the new OpenAI client.
# It will read your API key from the OPENAI_API_KEY environment variable.
client = OpenAI(api_key="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")


# ─── Transcription ───────────────────────────────────────────────────────────────
whisper_model = whisper.load_model(WHISPER_MODEL)

def transcribe_file(audio_path: str) -> str:
    """Transcribe a single audio file and return the transcript text."""
    result = whisper_model.transcribe(audio_path)
    return result["text"]


# ─── Insight Extraction ─────────────────────────────────────────────────────────
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

def extract_insights(transcript: str) -> dict:
    """Call OpenAI Chat API to extract structured insights from transcript."""
    messages = [
        {"role": "system", "content": "You extract structured call insights as JSON."},
        {"role": "user",   "content": INSIGHT_PROMPT.format(transcript=transcript)}
    ]
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.0,
        max_tokens=500
    )

    content = resp.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"error": "Could not parse JSON", "raw": content}


# ─── Pipeline Runner ────────────────────────────────────────────────────────────
def process_call(audio_path: str):
    """Full process: transcribe, save transcript, extract insights, save JSON."""
    fname = Path(audio_path).stem
    out_folder = Path(OUTPUT_DIR) / fname
    out_folder.mkdir(parents=True, exist_ok=True)

    # 1) Transcribe
    transcript = transcribe_file(audio_path)
    txt_path = out_folder / f"{fname}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    # 2) Extract Insights
    insights = extract_insights(transcript)
    json_path = out_folder / f"{fname}_insights.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(insights, f, ensure_ascii=False, indent=2)

    print(f"[Done] {fname} → transcript + insights")
    return fname


def main():
    audio_files = glob.glob(os.path.join(AUDIO_DIR, "*.*"))
    print(f"Found {len(audio_files)} audio files in {AUDIO_DIR!r}")
    for path in audio_files:
        process_call(path)
    # with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    #     futures = {executor.submit(process_call, path): path for path in audio_files}
    #     for future in as_completed(futures):
    #         path = futures[future]
    #         try:
    #             future.result()
    #         except Exception as exc:
    #             print(f"[Error] {path}: {exc!r}")


if __name__ == "__main__":
    main()
