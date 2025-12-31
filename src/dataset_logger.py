# src/dataset_logger.py
import os
import pandas as pd

AI_OUTPUT_PATH = "data/datasets/ai_outputs.xlsx"

AI_OUTPUT_COLUMNS = [
    "file_name",
    "company_name",
    "ceo_name",
    "bm",
    "industry",
    "stage",
    # 9 sections (0~5)
    "score_problem",
    "score_solution",
    "score_market",
    "score_business_model",
    "score_competition",
    "score_growth",
    "score_team",
    "score_finance",
    "score_risk",
    # totals
    "ai_raw_total_score",
    "ai_recommend_raw",   # YES/NO (>=80 기준)
    # meta
    "model_name",
    "prompt_version",
]

def ensure_dir():
    os.makedirs(os.path.dirname(AI_OUTPUT_PATH), exist_ok=True)

def load_ai_outputs() -> pd.DataFrame:
    ensure_dir()
    if os.path.exists(AI_OUTPUT_PATH):
        df = pd.read_excel(AI_OUTPUT_PATH)
    else:
        df = pd.DataFrame(columns=AI_OUTPUT_COLUMNS)

    # 컬럼 보정
    for c in AI_OUTPUT_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df = df[AI_OUTPUT_COLUMNS]
    return df

def upsert_ai_output(row: dict) -> None:
    """
    file_name을 키로 upsert:
    - 같은 file_name이 있으면 해당 행 업데이트
    - 없으면 신규 행 추가
    """
    ensure_dir()
    df = load_ai_outputs()

    # row 컬럼 보정
    new_row = {c: row.get(c, "") for c in AI_OUTPUT_COLUMNS}

    key = str(new_row["file_name"]).strip()
    if not key:
        raise ValueError("file_name이 비었습니다. upsert 불가")

    # upsert
    mask = df["file_name"].astype(str).str.strip() == key
    if mask.any():
        df.loc[mask, :] = pd.DataFrame([new_row]).iloc[0].values
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_excel(AI_OUTPUT_PATH, index=False)
