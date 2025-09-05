# habilitar pacote 'src' (se ainda não existir)
touch src/__init__.py

# treino (1 época por padrão)
python -m src.train

# exportar ONNX e testar inferências
python -m src.export_onnx
python src/infer.py --text "Breaking news: test"
python src/infer_onnx.py --text "Breaking news: test"
