import os
import re
import unicodedata
import zipfile
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModel, AutoConfig

_LEET = str.maketrans({"4": "a", "0": "o", "3": "e", "1": "i", "5": "s", "7": "t", "8": "b"})


def remove_vietnamese_accents(text: str) -> str:
    if not text:
        return text
    nfd = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return stripped.replace("đ", "d").replace("Đ", "D")


def deobfuscate_bypass(text: str) -> str:
    if not text:
        return text
    s = re.sub(r"\s+\.\s+", " ", text)
    s = re.sub(r"(?<=\S)\s*\.\s*(?=\S)", "", s)
    s = re.sub(r"\*+", "", s)
    s = s.translate(_LEET)
    return re.sub(r"\s+", " ", s).strip()

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_BASE = "vinai/phobert-base-v2"
GDRIVE_FILE_ID = "1rTAoeaSzIFu9okwCrA9dBOdR2FyN4Wsu"
GDRIVE_ZIP_PATH = os.path.join(MODEL_PATH, "model_download.zip")

KEYWORDS = []


def load_keyword_list():
    path = os.path.join(DATASET_DIR, "keywords.txt")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    seen = set()
    out = []
    for line in lines:
        for variant in (line, remove_vietnamese_accents(line)):
            key = variant.lower()
            if key and key not in seen:
                seen.add(key)
                out.append(variant)
    return out


def keyword_hits(text: str) -> list[str]:
    if not KEYWORDS or not text:
        return []
    lower = text.lower()
    plain = remove_vietnamese_accents(lower)
    deob = deobfuscate_bypass(lower)
    deob_plain = remove_vietnamese_accents(deob)
    haystacks = (lower, plain, deob, deob_plain)
    hits = []
    for kw in KEYWORDS:
        kw_l = kw.lower()
        kw_p = remove_vietnamese_accents(kw_l)
        if any(kw_l in h or kw_p in h for h in haystacks):
            hits.append(kw)
    return list(dict.fromkeys(hits))

class PhoBERTViolation(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.phobert = AutoModel.from_pretrained(model_name, low_cpu_mem_usage=True)
        hidden = self.config.hidden_size
        self.violation_head = nn.Linear(hidden, 2)
        self.keyword_head = nn.Linear(hidden, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        cls_out = self.dropout(outputs.pooler_output)
        seq_out = self.dropout(outputs.last_hidden_state)
        return self.violation_head(cls_out), self.keyword_head(seq_out)

def download_model_if_needed():
    bin_path = os.path.join(MODEL_PATH, "pytorch_model.bin")
    if os.path.exists(bin_path):
        return True
    os.makedirs(MODEL_PATH, exist_ok=True)
    try:
        import gdown
    except ImportError:
        os.system("pip install gdown -q")
        import gdown
    try:
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, GDRIVE_ZIP_PATH, quiet=True)
        if zipfile.is_zipfile(GDRIVE_ZIP_PATH):
            with zipfile.ZipFile(GDRIVE_ZIP_PATH, "r") as z:
                z.extractall(MODEL_PATH)
            os.remove(GDRIVE_ZIP_PATH)
        else:
            os.rename(GDRIVE_ZIP_PATH, bin_path)
        return True
    except:
        return False

tokenizer = None
model = None

if download_model_if_needed():
    try:
        bin_path = os.path.join(MODEL_PATH, "pytorch_model.bin")
        tok_path = MODEL_PATH if os.path.exists(os.path.join(MODEL_PATH, "tokenizer_config.json")) else MODEL_BASE
        tokenizer = AutoTokenizer.from_pretrained(tok_path)
        model = PhoBERTViolation(MODEL_BASE)
        model.load_state_dict(torch.load(bin_path, map_location=torch.device("cpu")))
        model.eval()
        import gc
        gc.collect()
    except:
        pass

KEYWORDS = load_keyword_list()

@app.route("/")
def index():
    return jsonify({
        "status": "online",
        "service": "AI Content Moderation",
        "endpoints": {"moderation": "/api/moderation"}
    })

@app.route("/api/moderation", methods=["POST"])
def predict():
    if not model or not tokenizer:
        return jsonify({"error": "Model not loaded"}), 500
    data = request.json
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400
    try:
        model_text = deobfuscate_bypass(text) or text
        inputs = tokenizer(model_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            v_logits, k_logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            probs = torch.nn.functional.softmax(v_logits, dim=-1)[0]
            pred_class = v_logits.argmax(dim=-1).item()
            confidence = probs[pred_class].item()
            is_violation = pred_class == 1
            label = "Vi phạm" if is_violation else "Không vi phạm"
            score = round(probs[1].item() * 100, 2)
            kw_preds = k_logits.argmax(dim=-1)[0]
            ids = inputs["input_ids"][0]
            spec = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
            kws = []
            for i, p in enumerate(kw_preds):
                if p == 1 and ids[i].item() not in spec:
                    t = tokenizer.decode([ids[i]]).replace("@@", "").strip()
                    if t and t not in kws: kws.append(t)
            if confidence < 0.55:
                label, is_violation, score, kws = "Không vi phạm", False, 0.0, []
            rule_kws = keyword_hits(text)
            if rule_kws:
                kws = list(dict.fromkeys(kws + rule_kws))
                if not is_violation or score < 50:
                    is_violation = True
                    label = "Vi phạm"
                    score = max(score, 85.0)
        return jsonify({
            "label": label,
            "is_violation": is_violation,
            "confidence": round(confidence * 100, 2),
            "violation_score": score,
            "keywords": kws
        })
    except:
        return jsonify({"error": "Internal error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
