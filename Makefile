PY=.venv/bin/python
setup:
	uv venv --python 3.12 .venv && uv pip install --python $(PY) -r requirements.txt
data:
	$(PY) scripts/download_data.py
test:
	$(PY) -m pytest -q
baselines:
	$(PY) scripts/run_baselines.py
final:
	$(PY) scripts/run_final.py
