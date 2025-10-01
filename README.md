# Teste AutoU — Classificador de E-mails com Flask + IA


## 🧩 Estrutura do Projeto
Teste-AutoU/├── backend/
│ ├── pp.py ← API Flask (endpoints /health e /classify)
│ ├── train_classifier.py ← script para treinar modelo local
│ ├── pipeline_email_classifier.joblib ← modelo salvo (após treino)
│ └── requirements.txt ← dependências Python
├── frontend/
│ └── index.html ← interface web HTML5 + JS


## Principais Funcionalidades
- Upload de e-mail em `.txt` ou `.pdf`, ou inserção direta de texto  
- Extração de texto (para arquivos)  
- Pré-processamento (remoção de stopwords, lematização, etc.)  
- Classificação binária: **Produtivo** ou **Improdutivo**  
- Geração de resposta automática (via fallback local ou via API de IA, se configurada)  
- Interface web simples que consome o backend  

##  Como rodar localmente


 1. Clonar o repositório
```bash
git clone https://github.com/Joaok25/Teste-AutoU.git
cd Teste-AutoU/backend

2. Criar ambiente virtual e instalar dependências
python -m venv venv
# No Linux/macOS:
source venv/bin/activate
# No Windows (PowerShell):
venv\Scripts\Activate.ps1

pip install -r requirements.txt


3. Treinar o modelo
python train_classifier.py

4. Iniciar o backend Flask
python app.py

5. Executar a interface frontend

Abra o arquivo frontend/index.html no navegador.
Ou, para evitar problemas de CORS, execute:
cd frontend
python -m http.server 8000

