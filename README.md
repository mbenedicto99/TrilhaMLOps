# MLOps — Trilha Mínima (HF + Lightning + Hydra + DVC local + ONNX + FastAPI + Docker + CI)

Este repositório é o **esqueleto mínimo** para:
1) Treinar um modelo de classificação de texto (AG News) com HuggingFace + PyTorch Lightning;
2) Gerenciar configs com **Hydra**;
3) (Opcional) Versionar dados localmente com **DVC**;
4) Exportar para **ONNX** e fazer inferência com **ONNXRuntime**;
5) Servir via **FastAPI**;
6) Empacotar em **Docker**;
7) Rodar **CI** simples no GitHub Actions (lint + build).

> Observação: W&B é opcional. Se não setar `WANDB_API_KEY` o script usa `CSVLogger`.

## Requisitos rápidos
- Python 3.10+
- pip, venv
- (Opcional) Docker
- (Opcional) DVC (local)

## Setup local
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# (Opcional) wandb login
```

## Treino (rápido)
```bash
python src/train.py
```

## Exportar ONNX
```bash
python src/export_onnx.py
```

## Inferência (PyTorch & ONNX)
```bash
python src/infer.py --text "Breaking news: test the pipeline."
python src/infer_onnx.py --text "Breaking news: test the pipeline."
```

## API FastAPI (usa ONNXRuntime)
```bash
uvicorn api.app:app --reload
# POST localhost:8000/predict  body: {"text": "some text"}
```

## Docker
```bash
docker build -t mlops-min:dev -f docker/Dockerfile .
docker run --rm -p 8000:8000 -e MODEL_PATH=/app/artifacts/model.onnx mlops-min:dev
# POST em http://localhost:8000/predict
```

## DVC (opcional, local)
```bash
dvc init
# Exemplo: versionar a pasta data/ (se você salvar artefatos lá)
dvc add data
git add data.dvc .dvc .dvcignore
git commit -m "track data locally with DVC"
```

## CI (GitHub Actions)
- Arquivo em `.github/workflows/ci.yml` faz lint (flake8) e build da imagem Docker.
- Ajuste conforme necessário (ex.: para publicar no GHCR/ECR).

## Estrutura
```
.
├─ src/
│  ├─ data_utils.py       # dataset & dataloaders (AG News)
│  ├─ model.py            # LightningModule (HF)
│  ├─ train.py            # treino com Hydra + W&B opcional
│  ├─ infer.py            # inferência PyTorch
│  ├─ export_onnx.py      # exportar modelo p/ ONNX
│  └─ infer_onnx.py       # inferência ONNXRuntime
├─ api/
│  └─ app.py              # FastAPI com endpoint /predict (ONNX)
├─ configs/
│  └─ config.yaml         # Hiperparâmetros (Hydra)
├─ docker/
│  └─ Dockerfile
├─ .github/workflows/
│  └─ ci.yml
├─ artifacts/             # onde ficam pesos treinados/onnx (gitignored)
├─ data/                  # dados locais (gitignored)
├─ requirements.txt
└─ README.md
```

## Notas de qualidade
- `ruff`/`flake8`/`black` são bem-vindos, aqui só deixamos `flake8` no CI para reduzir fricção.
- Seeds fixados no treino para favorecer reprodutibilidade mínima.
