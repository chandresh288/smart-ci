# Smart Call Analyzer: Speech Analytics for Agent Performance and KPI Tracking

## 📌 Overview
The **Smart Call Analyzer** is an AI-powered platform designed to automatically extract actionable insights from business/customer support call recordings.  
It transcribes calls, summarizes discussions, identifies key topics and action items, and detects sentiment — enabling **data-driven decision making** while reducing manual review efforts by over 80%.

This project was developed as part of the **Master of Data Science and Artificial Intelligence (M.Sc. DS&AI)** program at **Himachal Pradesh University, Shimla (2023–2025 batch).**

---

## 🚀 Features
- **High-Quality Transcription** using [OpenAI Whisper](https://github.com/openai/whisper) (handles Hindi, English, and Hinglish).
- **Insight Extraction** via GPT-4o:
  - Summaries (2–3 sentences per call)
  - Key discussion topics
  - Action items (with roles & deadlines)
  - Sentiment classification
  - Role classification (Agent vs Customer)
- **Analytics & Visualization** with Python (pandas + matplotlib):
  - Sentiment distribution
  - Action-item histograms
  - Top 10 frequent topics
- **Explainable AI (XAI)** using SHAP to make model predictions interpretable.
- **Cloud Deployment** on AWS:
  - S3 for storage
  - AWS Glue for ETL workflows
  - CloudFormation for Infrastructure-as-Code (IaC)

---

## 🏗️ System Architecture
1. **Audio Ingestion** (WAV/MP3 calls uploaded to S3)  
2. **Transcription** using Whisper ASR (PyTorch-based)  
3. **Insight Extraction** with GPT-4o → JSON outputs  
4. **Analytics & Visualization** (Python scripts generate CSVs and PNG charts)  
5. **Cloud Deployment** automated with AWS Glue & CloudFormation  

---

## 📊 Results (Highlights)
- **Word Error Rate (WER):** 8.2% on Hinglish calls  
- **Topic Extraction F1-Score:** 0.99  
- **Action Item Detection F1-Score:** 0.85  
- **Sentiment Classification Accuracy:** 0.97  
- Dashboards show:
  - ~87% of calls neutral in sentiment
  - Top topics: *technical support, camera configuration, Wi-Fi connectivity*
  - ~28% of calls involve ≥2 action items  

---

## 🛠️ Tech Stack
**Languages & Libraries**  
- Python 3.9+  
- pandas, numpy, matplotlib, glob, json, jiwer  
- PyTorch, Whisper (ASR), Hugging Face Transformers, OpenAI GPT-4o  
- SHAP (Explainable AI)  

**Cloud & Dev Tools**  
- AWS S3, AWS Glue, AWS CloudFormation  
- Git, Jupyter Notebook, VS Code  

**Hardware (for local experiments)**  
- Intel i7 CPU, NVIDIA RTX 3060 GPU, 32 GB RAM  

---

## 📂 Repository Structure
```

smart-ci/
│── data/                # Sample audio/transcripts (if available)
│── scripts/
│    ├── transcription.py     # Whisper ASR pipeline
│    ├── report\_generation.py # Aggregates insights & generates reports
│    └── test.py              # Evaluation and parallel execution
│── notebooks/
│    └── test.ipynb           # SHAP/XAI experiments
│── reports/             # Generated CSVs, PNG charts, SHAP visualizations
│── cloud/
│    └── workflow-call-analytics-pipeline.yaml  # CloudFormation template
│── README.md

````

---

## ⚡ Installation & Usage
### 1. Clone the Repository
```bash
git clone https://github.com/chandresh288/smart-ci.git
cd smart-ci
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Transcription

```bash
python scripts/transcription.py --input data/audio --output data/transcripts
```

### 4. Generate Reports

```bash
python scripts/report_generation.py --input data/transcripts --output reports/
```

### 5. View Dashboards

Check the `reports/` folder for sentiment distribution, topic frequency, and action-item charts.

---

## ☁️ AWS Deployment

* Upload raw call recordings to **S3**
* Deploy using **CloudFormation** template in `cloud/`
* AWS Glue runs transcription and report generation jobs every 15 minutes
* Outputs stored in `s3://<bucket>/reports/`

---

## 🔮 Future Work

* Real-time streaming pipeline for live insights
* Multilingual support & domain adaptation
* Lightweight on-prem deployment (optimized models)
* Enhanced explainability with counterfactual explanations
* Full-featured **Streamlit/React dashboard** for business users

---

## 📜 License

This project is for **academic and research purposes**. For enterprise use, please contact the author.

---

## 👩‍🎓 Author

**Chandresh Kumari**
M.Sc. Data Science & Artificial Intelligence (2023–2025)
Himachal Pradesh University, Shimla

---

Would you like me to also **add badges (Python version, AWS, license, etc.) and a project logo/screenshot** section to make your README look more professional for GitHub?
```
