from pathlib import Path

import pandas as pd


def export_to_csv(data: dict, filename: str = "resultados.csv") -> Path:
    df = pd.DataFrame(data)
    path = Path(filename)
    df.to_csv(path, index=False)
    return path


def export_to_latex(data: dict, filename: str = "resultados.tex") -> Path:
    df = pd.DataFrame(data)
    path = Path(filename)
    path.write_text(df.to_latex(index=False))
    return path
