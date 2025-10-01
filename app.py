import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from PyPDF2 import PdfReader
import spacy
from werkzeug.utils import secure_filename


# Open IA
try:
    import openai
except Exception:
    openai = None
    
    
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
CORS(app)  #permitir Frontend acessar o Backend 

MODEL_FILE = "pipeline_email_classifier.joblib"

try: 
    nlp = spacy.load("pt_core_news_sm")
    logging.info("spaCy pt model carregado")
except Exception:
    nlp = None
    logging.warning("spaCy pt_core_news_sm não encontrado. Rode: python -m spacy download pt_core_news_sm")
    
    
#pipeline
pipeline = None
if os.path.exists(MODEL_FILE):
    pipeline = joblib.load(MODEL_FILE)
    logging.info(f"Modelo carregado de {MODEL_FILE}")
else:
    logging.warning (f"Modelo {MODEL_FILE} não encontrado. Rode train_classifier.py para gerar o modelo.")
    

#Openai key
openai_KEY = os.getenv("OPENAI_API_KEY")
if openai and openai_KEY:
    openai.api_key = openai_KEY
    logging.info("OpenAI API key carregada")
else:
    if not openai_KEY:
        logging.info("OPENAI_API_KEY não configurada — será usado fallback para respostas")
    else:
        logging.info("OpenAI não disponível")
        

def preprocess(text:str) -> str:
    """Preprocessa o texto removendo stopwords e pontuação."""
    if not text:
        return ""
    if nlp is None:
        tokens = [t.lower() for t in text.split() if t.isalpha()]
        return " ".join(tokens)
    doc = nlp(text)
    tokens = [tok.lemma_.lower() for tok in doc if tok.is_alpha and not tok.is_stop]
    return " ".join(tokens)


def extract_text_from_file(uploaded_file):
    """Extrai texto de arquivos PDF ou TXT."""
    filename = secure_filename(uploaded_file.filename or "")
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    uploaded_file.stream.seek(0)
    if ext == "pdf":
        reader = PdfReader(uploaded_file.stream)
        pages = []
        for p in reader.pages:
            t = p.extract_text()
            if t:
                pages.append(t)
        return "\n".join(pages)
    elif ext == "txt":
        raw  = uploaded_file.read()
        try: 
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin-1", errors="ignore")
    else:
        try:
            raw  = uploaded_file.read()
            return raw.decode("utf-8",errors="ignore")
        except Exception:
            return ""
        
        

def predict_category(text: str) -> str:
    if pipeline is None:
        return{"category": "unknown", "confidence": 0.0}
    try:
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba([text])[0]
            classes = pipeline.classes_
            idx =int(probs.argmax())
            return {"category": classes[idx], "confidence": float(probs[idx])}
        else:
            cat = pipeline.predict([text])[0]
            return {"category": cat, "confidence": 1.0}
    except Exception as e:
        logging.error(f"Erro na predição: {e}")
        cat =pipeline.predict([text])[0] if pipeline else "unknown"
        return {"category": str(cat), "confidence": 0.0}
    
    
    
def fallback_reply(category:str, email_text:str ="")-> str:
    if category == "Produtivo":
        return "Obrigado pelo seu email. Vamos analisar sua solicitação e responderemos em breve."
    elif category == "Improdutivo":
        return "Agradecemos seu contato. Desejamos a você boas festas!"
    else:
        return "Obrigado pelo seu email. Entraremos em contato em breve."
    
    
def generate_reply_openai(email_text:str, category:str) -> str:
    if not openai_KEY or not openai:
        raise RuntimeError("OpenAI não disponível")
    prompt = f"""esponda de forma profissional e educada o seguinte email classificado como '{category}'.
Email:
\"\"\"{email_text}\"\"\"

Escreva uma resposta curta, profissional e direta (breve saudação, resumo dos proximos passos). Máx 6 linhas."""

    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Você é um assistente profissional."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=250,
        temperature=0.2
    )
    return resp["choices"][0]["message"]["content"].strip()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": bool(pipeline),
        "nlp_loaded": bool(nlp)
    })
    

@app.route("/classify", methods=["POST"])
def classify():
    #receve texto direto
    text = request.form.get("text", "")
    file = request.files.get("file")
    if file and file.filename:
        text = extract_text_from_file(file)
        
    if not text or not text.strip():
        return jsonify({"error": "Nenhum texto fornecido"}), 400
    
    clean = preprocess(text)
    pred = predict_category(clean)
    category = pred["category"]
    confidence = pred["confidence"]
    
    #RESPOSTA
    try:
        if openai and openai_KEY is not None:
            reply = generate_reply_openai(text, category)
        else:
            reply = fallback_reply(category, text)
    except Exception as e:
        logging.error(f"Erro ao gerar resposta com OpenAI: {e}")
        reply = fallback_reply(category, text)
        
    return jsonify({
        "category": category,
        "confidence": confidence,
        "suggested_reply": reply
    })
    
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)