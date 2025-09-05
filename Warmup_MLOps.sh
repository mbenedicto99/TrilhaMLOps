# 0) Vá para a raiz do projeto
cd "/home/mbenedicto/Documents/Faculdades/POLI/IA/trilhaMLOps"

# 1) Garanta o módulo de venv
sudo apt update && sudo apt install -y python3.12-venv

# 2) Crie/renove o venv e ative
python3.12 -m venv .venv --upgrade-deps
source .venv/bin/activate

# 3) Confirme que está no venv
which python
python - <<'PY'
import sys; print("venv?", sys.prefix != sys.base_prefix, "|", sys.executable)
PY

# 4) Instale o PyTorch primeiro (CPU). Depois o resto.
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
pip install -r requirements.txt

# 5) (se ainda não existir) habilite pacote src
touch src/__init__.py

# 6) Smoke test de imports
python - <<'PY'
import torch, pytorch_lightning, onnxruntime, transformers, datasets
print("OK:",
      "torch", torch.__version__,
      "| pl", pytorch_lightning.__version__,
      "| ort", onnxruntime.__version__)
PY

# 7) Treino / export / inferência
python -m src.train
python -m src.export_onnx
python src/infer.py --text "Breaking news: test"
python src/infer_onnx.py --text "Breaking news: test"

