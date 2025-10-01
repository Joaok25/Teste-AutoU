import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
data = {
    "text": [
        "Preciso de ajuda com minha conta, não consigo acessar.",
        "Qual é o status do minha solicitação?",
        "Feliz natal para todos!",
        "Segue em anexo",
        "Apenas passando para desejar boas festas"
    ],
    "label": [
        "Produtivo",
        "Produtivo",
        "Improdutivo",
        "Produtivo",
        "Improdutivo"
    ]
}

df = pd.DataFrame(data)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=1000))
])

pipeline.fit(df["text"], df["label"])
joblib.dump(pipeline, "pipeline_email_classifier.joblib")
print("Modelo salvo em pipeline_email_classifier.joblib")
    