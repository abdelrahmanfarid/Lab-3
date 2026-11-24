"""Execute the lab notebook with the project virtual environment."""
import json
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / '.venv/bin/python'
SPEC = ROOT / '.venv/share/jupyter/kernels/lab3-venv'
SPEC.mkdir(parents=True, exist_ok=True)
(SPEC / 'kernel.json').write_text(json.dumps({
    'argv':[str(PYTHON), '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name':'Lab 3 virtual environment', 'language':'python'}))
os.environ['JUPYTER_PATH'] = str(ROOT / '.venv/share/jupyter')
path = ROOT / 'lab3_tasks_1_2_3.ipynb'
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=3600, kernel_name='lab3-venv',
                        resources={'metadata':{'path':str(ROOT)}},
                        on_cell_start=lambda cell, cell_index: print(f'Starting cell {cell_index + 1}/{len(nb.cells)}', flush=True),
                        on_cell_complete=lambda cell, cell_index: print(f'Completed cell {cell_index + 1}/{len(nb.cells)}', flush=True))
client.execute()
nbformat.write(nb, path)
print('Executed and saved', path.name)
