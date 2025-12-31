import os
import pandas as pd
from datetime import datetime

HISTORY_PATH = "data/history/history.xlsx"

def append_history(row: dict):
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)

    row = dict(row)
    row["timestamp"] = row.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if os.path.exists(HISTORY_PATH):
        df = pd.read_excel(HISTORY_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_excel(HISTORY_PATH, index=False)

def load_history():
    if os.path.exists(HISTORY_PATH):
        return pd.read_excel(HISTORY_PATH)
    return pd.DataFrame(columns=[
        "timestamp","company_name","ceo_name","bm","industry","stage",
        "total_score","recommendation","file_name","output_path"
    ])
