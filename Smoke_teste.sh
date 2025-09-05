# DIR 
cd "/home/mbenedicto/Documents/Faculdades/POLI/IA/trilhaMLOps"

# Atualiza VENV
sudo apt update && sudo apt install -y python3.12-venv

python3.12 -m venv .venv --upgrade-deps
source .venv/bin/activate

# Teste VENV
which python
python - <<'PY'
import sys; print("venv?", sys.prefix != sys.base_prefix, "|", sys.executable)
PY

# Atualiza PyTorch
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
pip install -r requirements.txt

# Habilita SRC
touch src/__init__.py

# Importa SMOKE TEST
python - <<'PY'
import torch, pytorch_lightning, onnxruntime, transformers, datasets
print("OK:",
      "torch", torch.__version__,
      "| pl", pytorch_lightning.__version__,
      "| ort", onnxruntime.__version__)
PY

# Treina, analise e exibe resultado
python -m src.train
python -m src.export_onnx
python src/infer.py --text "Breaking news: test"
python src/infer_onnx.py --text "Breaking news: test"

