# src/presets.py
from typing import Dict

EVAL_ITEMS = [
    "회사 개요",
    "문제 정의",
    "솔루션",
    "시장 분석",
    "비즈니스 모델",
    "경쟁 분석",
    "트랙션",
    "성장전략",
    "팀 구성",
    "재무 계획",
]

BASE_WEIGHTS: Dict[str, float] = {
    "회사 개요": 0.08,
    "문제 정의": 0.10,
    "솔루션": 0.12,
    "시장 분석": 0.12,
    "비즈니스 모델": 0.12,
    "경쟁 분석": 0.10,
    "트랙션": 0.12,
    "성장전략": 0.10,
    "팀 구성": 0.07,
    "재무 계획": 0.07,
}

BM_PRESETS: Dict[str, Dict[str, float]] = {
    "SaaS": {"트랙션": 0.14, "성장전략": 0.12, "비즈니스 모델": 0.13, "재무 계획": 0.06},
    "플랫폼": {"시장 분석": 0.14, "비즈니스 모델": 0.13, "트랙션": 0.13, "경쟁 분석": 0.11},
    "제조": {"솔루션": 0.13, "비즈니스 모델": 0.13, "재무 계획": 0.10, "팀 구성": 0.08},
    "딥테크": {"솔루션": 0.14, "시장 분석": 0.13, "팀 구성": 0.09, "트랙션": 0.10},
    "바이오": {"솔루션": 0.14, "시장 분석": 0.13, "재무 계획": 0.10, "팀 구성": 0.08},
}

STAGE_PRESETS: Dict[str, Dict[str, float]] = {
    "Pre-seed": {"팀 구성": 0.10, "문제 정의": 0.12, "솔루션": 0.14, "트랙션": 0.08, "재무 계획": 0.05},
    "Seed": {"팀 구성": 0.09, "문제 정의": 0.11, "솔루션": 0.13, "트랙션": 0.10, "재무 계획": 0.06},
    "Series A": {"트랙션": 0.14, "성장전략": 0.12, "비즈니스 모델": 0.13, "재무 계획": 0.08},
    "Series B+": {"트랙션": 0.15, "재무 계획": 0.10, "성장전략": 0.12, "경쟁 분석": 0.11},
}

def normalize(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values()) if weights else 1.0
    if total == 0:
        return weights
    return {k: v / total for k, v in weights.items()}

def merge_presets(bm: str, stage: str) -> Dict[str, float]:
    w = dict(BASE_WEIGHTS)
    if bm in BM_PRESETS:
        for k, v in BM_PRESETS[bm].items():
            w[k] = v
    if stage in STAGE_PRESETS:
        for k, v in STAGE_PRESETS[stage].items():
            w[k] = v
    return normalize(w)
