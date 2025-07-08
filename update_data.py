import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

def run_notebook(notebook_path):
    """Ejecuta el notebook y guarda los outputs"""
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
    
    # Guardar el notebook ejecutado (opcional)
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == "__main__":
    notebook_path = "notebooks/Support Vector Machine Counter Strike.ipynb"
    run_notebook(notebook_path)
    print("✅ Notebook ejecutado y datos exportados")