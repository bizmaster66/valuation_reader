# src/presets_simple.py
from typing import Dict

# 엑셀 평가 섹션(9개)
SECTIONS = [
    "문제 정의",
    "솔루션 & 제품",
    "시장 분석",
    "비즈니스 모델",
    "경쟁 우위",
    "성장 전략",
    "팀 역량",
    "재무 계획",
    "리스크 관리",
]

BASE_WEIGHTS: Dict[str, float] = {
    "문제 정의": 0.12,
    "솔루션 & 제품": 0.13,
    "시장 분석": 0.13,
    "비즈니스 모델": 0.12,
    "경쟁 우위": 0.10,
    "성장 전략": 0.12,
    "팀 역량": 0.10,
    "재무 계획": 0.10,
    "리스크 관리": 0.08,
}

BM_PRESETS: Dict[str, Dict[str, float]] = {
    "SaaS": {"성장 전략": 0.14, "비즈니스 모델": 0.13, "재무 계획": 0.08, "리스크 관리": 0.06},
    "플랫폼": {"시장 분석": 0.14, "비즈니스 모델": 0.13, "경쟁 우위": 0.11},
    "제조": {"솔루션 & 제품": 0.14, "재무 계획": 0.12, "리스크 관리": 0.10},
    "딥테크": {"솔루션 & 제품": 0.15, "시장 분석": 0.14, "팀 역량": 0.11},
    "바이오": {"솔루션 & 제품": 0.15, "시장 분석": 0.14, "리스크 관리": 0.10},
    "커머스": {"비즈니스 모델": 0.13, "성장 전략": 0.13, "경쟁 우위": 0.11},
}

STAGE_PRESETS: Dict[str, Dict[str, float]] = {
    "Pre-seed": {"문제 정의": 0.14, "솔루션 & 제품": 0.15, "팀 역량": 0.12, "재무 계획": 0.06},
    "Seed": {"문제 정의": 0.13, "솔루션 & 제품": 0.14, "성장 전략": 0.13, "재무 계획": 0.07},
    "Series A": {"성장 전략": 0.14, "비즈니스 모델": 0.13, "재무 계획": 0.10},
    "Series B+": {"성장 전략": 0.15, "재무 계획": 0.12, "경쟁 우위": 0.11},
}

def normalize(w: Dict[str, float]) -> Dict[str, float]:
    s = sum(w.values()) or 1.0
    return {k: v / s for k, v in w.items()}

def merge_presets(bm: str, stage: str) -> Dict[str, float]:
    w = dict(BASE_WEIGHTS)
    if bm in BM_PRESETS:
        w.update(BM_PRESETS[bm])
    if stage in STAGE_PRESETS:
        w.update(STAGE_PRESETS[stage])
    # 누락 섹션 보정
    for k in SECTIONS:
        w.setdefault(k, 0.0)
    return normalize(w)
