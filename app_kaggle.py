import os, subprocess, sys

def install_deps():
    if os.path.exists('/kaggle/working'):
        for lib in ['flask-cors', 'pyngrok', 'gdown']:
            try: __import__(lib.replace('-', '_'))
            except ImportError:
                subprocess.check_call([sys.executable, "-m", "pip", "install", lib, "-q"])

install_deps()

import json, zipfile, torch, threading, time, urllib.request, shutil, traceback, re, unicodedata
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

ON_KAGGLE = os.path.exists('/kaggle/working')
BASE_DIR = '/kaggle/working' if ON_KAGGLE else os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_BASE = "vinai/phobert-base-v2"
GDRIVE_FILE_ID = "1rTAoeaSzIFu9okwCrA9dBOdR2FyN4Wsu"
AUTH_ENV_TXT_KAGGLE = '/kaggle/input/datasets/fwepofp/auth-ngrok-moderation/env.txt'

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


def keyword_hits(text: str) -> list:
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

app = Flask(__name__)
CORS(app)

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

def load_env():
    env_path = AUTH_ENV_TXT_KAGGLE if ON_KAGGLE else os.path.join(BASE_DIR, 'env.txt')
    if not os.path.exists(env_path): return
    try:
        with open(env_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line: continue
                k, v = line.split('=', 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")
    except: pass

load_env()

def find_file_by_ext(root_path, ext):
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(ext): return os.path.join(root, file)
    return None

def download_model_if_needed():
    # Kiểm tra xem có file .bin nào trong MODEL_PATH chưa
    if find_file_by_ext(MODEL_PATH, ".bin"): return True
    os.makedirs(MODEL_PATH, exist_ok=True)
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        tmp_path = os.path.join(MODEL_PATH, "download_temp")
        gdown.download(url, tmp_path, quiet=False, fuzzy=True)
        if not os.path.exists(tmp_path): return False
        with open(tmp_path, 'rb') as f:
            header = f.read(100)
        if header.startswith(b'PK'):
            print("Phát hiện file ZIP, đang giải nén...")
            with zipfile.ZipFile(tmp_path, "r") as z: z.extractall(MODEL_PATH)
            os.remove(tmp_path)
        elif header.startswith(b'Rar!'):
            print("Phát hiện file RAR, đang giải nén...")
            subprocess.run(["unrar", "x", "-o+", tmp_path, MODEL_PATH])
            os.remove(tmp_path)
        elif b'<!DOCTYPE' in header or b'<html' in header:
            print("LỖI: Link Drive bị chặn.")
            os.remove(tmp_path)
            return False
        else:
            os.rename(tmp_path, os.path.join(MODEL_PATH, "pytorch_model.bin"))
        return True
    except Exception as e:
        print(f"Lỗi tải: {e}")
        return False

tokenizer, model = None, None

print("=======================================")
print(f"Đang tải Tokenizer và Model Binary (Kaggle Mode)...")
if download_model_if_needed():
    try:
        bin_path = find_file_by_ext(MODEL_PATH, ".bin")
        # Tìm thư mục chứa config (có thể nằm sâu bên trong)
        config_file = find_file_by_ext(MODEL_PATH, "tokenizer_config.json")
        tok_path = os.path.dirname(config_file) if config_file else MODEL_BASE
        
        print(f"Sử dụng Model File: {bin_path}")
        print(f"Sử dụng Tokenizer Path: {tok_path}")

        tokenizer = AutoTokenizer.from_pretrained(tok_path)
        model = PhoBERTViolation(MODEL_BASE)
        model.load_state_dict(torch.load(bin_path, map_location=torch.device("cpu"), weights_only=False))
        model.eval()
        import gc; gc.collect()
        print(">>> Tải mô hình THÀNH CÔNG!")
    except Exception as e:
        traceback.print_exc()
        print(f">>> LỖI khi tải mô hình: {e}")
print("=======================================")

KEYWORDS = load_keyword_list()

@app.route("/")
def index(): return jsonify({"status": "online", "service": "AI Moderation"})

@app.route("/api/moderation", methods=["POST"])
def predict():
    if not model or not tokenizer: return jsonify({"error": "Model not loaded"}), 500
    data = request.json
    text = data.get("text", "").strip()
    if not text: return jsonify({"error": "Empty text"}), 400
    try:
        model_text = deobfuscate_bypass(text) or text
        inputs = tokenizer(model_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            v_logits, k_logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            probs = torch.nn.functional.softmax(v_logits, dim=-1)[0]
            pred_class = v_logits.argmax(dim=-1).item()
            conf = probs[pred_class].item()
            is_viol = pred_class == 1
            label = "Vi phạm" if is_viol else "Không vi phạm"
            score = round(probs[1].item() * 100, 2)
            kw_preds, ids = k_logits.argmax(dim=-1)[0], inputs["input_ids"][0]
            spec = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
            kws = []
            for i, p in enumerate(kw_preds):
                if p == 1 and ids[i].item() not in spec:
                    t = tokenizer.decode([ids[i]]).replace("@@", "").strip()
                    if t and t not in kws: kws.append(t)
            if conf < 0.55: label, is_viol, score, kws = "Không vi phạm", False, 0.0, []
            rule_kws = keyword_hits(text)
            if rule_kws:
                kws = list(dict.fromkeys(kws + rule_kws))
                if not is_viol or score < 50:
                    is_viol = True
                    label = "Vi phạm"
                    score = max(score, 85.0)
        return jsonify({"label": label, "is_violation": is_viol, "confidence": round(conf * 100, 2), "violation_score": score, "keywords": kws})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def start_ngrok(port):
    try:
        from pyngrok import ngrok
        tok = (os.environ.get('AUTH_NGROK_MODERATION_APP') or os.environ.get('NGROK_AUTHTOKEN'))
        if not tok: return
        ngrok.set_auth_token(tok)
        tunnel = ngrok.connect(port)
        print(f"\nURL: {tunnel.public_url}\n")
    except: pass

if __name__ == "__main__":
    port = 5000
    if ON_KAGGLE:
        threading.Thread(target=lambda: app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False), daemon=True).start()
        time.sleep(5)
        start_ngrok(port)
        while True: time.sleep(60)
    else: app.run(host="0.0.0.0", port=port)
