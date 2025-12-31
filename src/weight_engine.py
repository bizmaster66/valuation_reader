# weight_engine.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Any
import json


# -----------------------------
# 1) 기준 축(당신의 9개 평가 항목)
# -----------------------------
CRITERIA = [
    "problem_definition",     # 문제 정의
    "solution_product",       # 솔루션 & 제품
    "market",                 # 시장 분석
    "business_model",         # 비즈니스 모델
    "competitive_advantage",  # 경쟁 우위
    "growth_strategy",        # 성장 전략
    "team",                   # 팀 역량
    "financial_plan",         # 재무 계획
    "risk_management",        # 리스크 관리
]


# -----------------------------
# 2) Stage(라운드) 기본 가중치 (합 100)
#    - 당신이 앞서 만든 테이블을 기본값으로 반영
# -----------------------------
DEFAULT_STAGE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "seed": {
        "problem_definition": 12, "solution_product": 16, "market": 16,
        "business_model": 10, "competitive_advantage": 8, "growth_strategy": 10,
        "team": 18, "financial_plan": 5, "risk_management": 5
    },
    "pre_a": {
        "problem_definition": 10, "solution_product": 15, "market": 15,
        "business_model": 12, "competitive_advantage": 8, "growth_strategy": 14,
        "team": 14, "financial_plan": 6, "risk_management": 6
    },
    "series_a": {
        "problem_definition": 8, "solution_product": 14, "market": 14,
        "business_model": 14, "competitive_advantage": 10, "growth_strategy": 16,
        "team": 12, "financial_plan": 6, "risk_management": 6
    },
    "series_b": {
        "problem_definition": 6, "solution_product": 10, "market": 12,
        "business_model": 16, "competitive_advantage": 12, "growth_strategy": 18,
        "team": 10, "financial_plan": 10, "risk_management": 6
    },
    "series_c": {
        "problem_definition": 4, "solution_product": 8, "market": 12,
        "business_model": 18, "competitive_advantage": 14, "growth_strategy": 16,
        "team": 8, "financial_plan": 12, "risk_management": 8
    },
    "pre_ipo": {
        "problem_definition": 3, "solution_product": 6, "market": 10,
        "business_model": 18, "competitive_advantage": 15, "growth_strategy": 12,
        "team": 8, "financial_plan": 18, "risk_management": 10
    },
    "ipo": {
        "problem_definition": 2, "solution_product": 5, "market": 10,
        "business_model": 16, "competitive_advantage": 15, "growth_strategy": 10,
        "team": 7, "financial_plan": 22, "risk_management": 13
    },
}

# 사용자가 입력하는 stage 문자열 정규화(동의어 처리)
STAGE_ALIASES = {
    "pre-seed": "seed",
    "preseed": "seed",
    "seed": "seed",
    "pre a": "pre_a",
    "pre-a": "pre_a",
    "pre_a": "pre_a",
    "series a": "series_a",
    "a": "series_a",
    "series_a": "series_a",
    "series b": "series_b",
    "b": "series_b",
    "series_b": "series_b",
    "series c": "series_c",
    "c": "series_c",
    "series_c": "series_c",
    "pre ipo": "pre_ipo",
    "pre-ipo": "pre_ipo",
    "pre_ipo": "pre_ipo",
    "ipo": "ipo",
}


# -----------------------------
# 3) Industry / Business Model 멀티플라이어
#    - 1.0이 기본, 특정 항목을 더 중요하게 보고 싶으면 >1.0
#    - 최종 weights = normalize(stage_weights * industry_mult * model_mult)
# -----------------------------
DEFAULT_INDUSTRY_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    # 바이오/헬스케어: 제품/근거(임상/규제) + 리스크/팀 + IP(경쟁우위) 강화
    "bio_healthcare": {
        "solution_product": 1.25,
        "competitive_advantage": 1.10,
        "team": 1.10,
        "risk_management": 1.15,
        "financial_plan": 0.90,
        "growth_strategy": 0.90,
    },
    # 딥테크: 제품(TRL/상용화), 리스크(인증/규제), 재무(Capex), 경쟁우위(IP/공정) 강화
    "deeptech": {
        "solution_product": 1.15,
        "competitive_advantage": 1.10,
        "financial_plan": 1.10,
        "risk_management": 1.10,
        "growth_strategy": 1.05,
        "problem_definition": 0.95,
    },
    # 소부장/제조: 경쟁우위(공정/원가/공급망), 재무(마진/원가구조), 성장전략(양산/채널), 리스크(품질/공급망)
    "manufacturing_sobu": {
        "competitive_advantage": 1.15,
        "financial_plan": 1.15,
        "growth_strategy": 1.10,
        "risk_management": 1.10,
        "solution_product": 1.05,
        "team": 0.95,
    },
    # 플랫폼/마켓플레이스: 성장전략(리퀴디티), BM(테이크레이트), 경쟁우위(네트워크) 강화
    "platform_marketplace": {
        "growth_strategy": 1.25,
        "business_model": 1.15,
        "competitive_advantage": 1.15,
        "solution_product": 0.95,
        "financial_plan": 0.90,
    },
    # B2B SaaS: 성장전략(채널/세일즈), 재무(유닛이코노믹스), BM(반복과금) 강화
    "b2b_saas": {
        "growth_strategy": 1.15,
        "financial_plan": 1.15,
        "business_model": 1.10,
        "solution_product": 1.05,
        "risk_management": 0.95,
    },
    # D2C/커머스: 재무(마진/재고/물류), BM(가격/유통), 성장전략(CAC/LTV) 강화
    "d2c_commerce": {
        "financial_plan": 1.20,
        "business_model": 1.15,
        "growth_strategy": 1.10,
        "solution_product": 1.00,
        "competitive_advantage": 1.05,
        "risk_management": 0.95,
    },
    # 핀테크/규제: 리스크(규제/보안), 팀(컴플라이언스), BM(라이선스/수수료) 강화
    "fintech_regulated": {
        "risk_management": 1.25,
        "team": 1.10,
        "business_model": 1.10,
        "solution_product": 1.00,
        "financial_plan": 1.05,
        "growth_strategy": 0.95,
    },
}

DEFAULT_BUSINESS_MODEL_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    # 구독형 SaaS: 재무/성장(리텐션/NRR/CAC payback) 더 강조
    "subscription_saas": {
        "financial_plan": 1.10,
        "growth_strategy": 1.10,
        "business_model": 1.05,
    },
    # usage-based: BM/재무(과금 구조, 변동원가) 더 강조
    "usage_based": {
        "business_model": 1.10,
        "financial_plan": 1.10,
        "growth_strategy": 1.05,
    },
    # transaction marketplace: 성장전략/리스크(사기/거래안전)/BM 강화
    "transaction_marketplace": {
        "growth_strategy": 1.15,
        "business_model": 1.10,
        "risk_management": 1.10,
    },
    # enterprise sales: 성장전략/팀(세일즈 실행력) 강화
    "enterprise_sales": {
        "growth_strategy": 1.10,
        "team": 1.05,
        "financial_plan": 1.05,
    },
}


# -----------------------------
# 4) Gate(게이트) 규칙
#    - 특정 항목이 너무 낮으면 overall 상한을 걸거나 패널티 부여
#    - "가중치"보다 실제 투자 심사에 가깝게 작동시키는 장치
# -----------------------------
@dataclass
class GateRule:
    name: str
    when: Dict[str, Any]                 # {"stage_min": "series_b", "industry_in": [...]} 등
    condition: Dict[str, Any]            # {"criterion": "financial_plan", "lt": 2.5}
    action: Dict[str, Any]               # {"cap_overall": 65} or {"penalty": 8}
    note: str = ""

STAGE_ORDER = ["seed", "pre_a", "series_a", "series_b", "series_c", "pre_ipo", "ipo"]
STAGE_RANK = {s: i for i, s in enumerate(STAGE_ORDER)}

DEFAULT_GATES: List[GateRule] = [
    # 규제산업/바이오에서 리스크 점수가 낮으면 큰 상한
    GateRule(
        name="regulated_risk_floor",
        when={"industry_in": ["fintech_regulated", "bio_healthcare"], "stage_min": "pre_a"},
        condition={"criterion": "risk_management", "lt": 2.5},
        action={"cap_overall": 65},
        note="규제/임상/보안 리스크가 낮게 평가되면 투자 논리 자체가 흔들릴 수 있음."
    ),
    # Series B 이상 SaaS에서 재무(유닛이코노믹스) 점수가 낮으면 패널티
    GateRule(
        name="saas_finance_bar_series_b_plus",
        when={"industry_in": ["b2b_saas"], "stage_min": "series_b"},
        condition={"criterion": "financial_plan", "lt": 3.0},
        action={"penalty": 8},
        note="Series B+에서는 NRR/CAC payback/효율 등 재무/성장 지표 기대치가 높아지는 경향."
    ),
    # 마켓플레이스에서 성장전략(리퀴디티) 낮으면 패널티
    GateRule(
        name="marketplace_liquidity_bar",
        when={"industry_in": ["platform_marketplace"], "stage_min": "series_a"},
        condition={"criterion": "growth_strategy", "lt": 3.0},
        action={"penalty": 6},
        note="마켓플레이스는 리퀴디티/네트워크 효과를 만들어내는 성장전략이 핵심."
    ),
]


# -----------------------------
# 5) 엔진 본체
# -----------------------------
def _normalize_weights(raw: Dict[str, float]) -> Dict[str, float]:
    # 누락 항목은 0으로 채우고, 음수 방지
    cleaned = {k: max(0.0, float(raw.get(k, 0.0))) for k in CRITERIA}
    total = sum(cleaned.values())
    if total <= 0:
        # 방어: 모두 0이면 균등
        eq = 100.0 / len(CRITERIA)
        return {k: eq for k in CRITERIA}
    return {k: (v / total) * 100.0 for k, v in cleaned.items()}


def _apply_multipliers(
    base: Dict[str, float],
    multipliers: Optional[Dict[str, float]],
) -> Dict[str, float]:
    if not multipliers:
        return dict(base)
    out = dict(base)
    for k, m in multipliers.items():
        if k in out:
            out[k] = out[k] * float(m)
    return out


def _rank_ok(stage: str, stage_min: Optional[str]) -> bool:
    if not stage_min:
        return True
    return STAGE_RANK[stage] >= STAGE_RANK[stage_min]


def _gate_applies(rule: GateRule, stage: str, industry: Optional[str], business_model: Optional[str]) -> bool:
    w = rule.when or {}
    if "stage_min" in w and not _rank_ok(stage, w["stage_min"]):
        return False
    if "industry_in" in w:
        if industry is None or industry not in set(w["industry_in"]):
            return False
    if "business_model_in" in w:
        if business_model is None or business_model not in set(w["business_model_in"]):
            return False
    return True


@dataclass
class WeightEngineConfig:
    stage_weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: DEFAULT_STAGE_WEIGHTS)
    industry_multipliers: Dict[str, Dict[str, float]] = field(default_factory=lambda: DEFAULT_INDUSTRY_MULTIPLIERS)
    business_model_multipliers: Dict[str, Dict[str, float]] = field(default_factory=lambda: DEFAULT_BUSINESS_MODEL_MULTIPLIERS)
    gates: List[GateRule] = field(default_factory=lambda: DEFAULT_GATES)

    @staticmethod
    def from_json(path: str) -> "WeightEngineConfig":
        """
        선택: 외부 JSON 설정 파일로 커스터마이즈하고 싶을 때 사용.
        JSON 예시:
        {
          "stage_weights": {...},
          "industry_multipliers": {...},
          "business_model_multipliers": {...},
          "gates": [...]
        }
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cfg = WeightEngineConfig()
        if "stage_weights" in data:
            cfg.stage_weights = data["stage_weights"]
        if "industry_multipliers" in data:
            cfg.industry_multipliers = data["industry_multipliers"]
        if "business_model_multipliers" in data:
            cfg.business_model_multipliers = data["business_model_multipliers"]
        if "gates" in data:
            gates: List[GateRule] = []
            for g in data["gates"]:
                gates.append(GateRule(**g))
            cfg.gates = gates
        return cfg


class WeightEngine:
    def __init__(self, config: Optional[WeightEngineConfig] = None):
        self.config = config or WeightEngineConfig()

    def normalize_stage(self, stage: str) -> str:
        s = (stage or "").strip().lower()
        s = s.replace("_", " ").replace("-", " ").strip()
        # alias 매칭
        if s in STAGE_ALIASES:
            return STAGE_ALIASES[s]
        # "series a" 등 공백 포함 케이스
        if s.replace(" ", "") in STAGE_ALIASES:
            return STAGE_ALIASES[s.replace(" ", "")]
        raise ValueError(f"Unknown stage: {stage}")

    def compute_weights(
        self,
        stage: str,
        industry: Optional[str] = None,
        business_model: Optional[str] = None,
        override_stage_weights: Optional[Dict[str, float]] = None,
        extra_multipliers: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        최종 가중치(합 100) 반환.
        - stage 기본 → industry multiplier → business_model multiplier → extra multiplier → normalize
        """
        st = self.normalize_stage(stage)
        base = self.config.stage_weights.get(st)
        if not base:
            raise ValueError(f"Stage weights missing for: {st}")

        raw = dict(base)

        # 사용자가 stage 기본값 자체를 덮어쓰고 싶은 경우
        if override_stage_weights:
            for k, v in override_stage_weights.items():
                if k in CRITERIA:
                    raw[k] = float(v)

        # industry multiplier
        if industry:
            raw = _apply_multipliers(raw, self.config.industry_multipliers.get(industry))

        # business model multiplier
        if business_model:
            raw = _apply_multipliers(raw, self.config.business_model_multipliers.get(business_model))

        # 추가 multiplier (실험/AB테스트용)
        raw = _apply_multipliers(raw, extra_multipliers)

        return _normalize_weights(raw)

    def score(
        self,
        scores_1_to_5: Dict[str, float],
        weights: Dict[str, float],
    ) -> float:
        """
        가중 평균 점수(1~5 스케일) 반환.
        """
        # 누락은 0 처리(원하면 여기서 예외로 바꿔도 됨)
        s = {k: float(scores_1_to_5.get(k, 0.0)) for k in CRITERIA}
        w = {k: float(weights.get(k, 0.0)) for k in CRITERIA}
        total_w = sum(w.values())
        if total_w <= 0:
            return 0.0
        weighted = sum(s[k] * w[k] for k in CRITERIA) / total_w
        return weighted

    def apply_gates(
        self,
        stage: str,
        industry: Optional[str],
        business_model: Optional[str],
        score_by_criterion: Dict[str, float],
        overall_1_to_5: float,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        gate 적용 후 overall(1~5) 점수 및 적용 내역 반환.
        action:
          - {"cap_overall": 65} => overall을 100점 환산 기준 cap (예: 65/100)로 제한
          - {"penalty": 8} => 100점 환산 기준 패널티 차감
        """
        st = self.normalize_stage(stage)
        applied: List[Dict[str, Any]] = []

        # overall을 100점으로 환산해서 캡/패널티 적용 후 다시 1~5로 환산
        overall_100 = (overall_1_to_5 / 5.0) * 100.0

        for rule in self.config.gates:
            if not _gate_applies(rule, st, industry, business_model):
                continue

            c = rule.condition
            crit = c.get("criterion")
            if crit not in CRITERIA:
                continue

            val = float(score_by_criterion.get(crit, 0.0))
            lt = c.get("lt")
            gte = c.get("gte")

            triggered = False
            if lt is not None and val < float(lt):
                triggered = True
            if gte is not None and val >= float(gte):
                triggered = True

            if not triggered:
                continue

            act = rule.action or {}
            before = overall_100
            if "cap_overall" in act:
                cap = float(act["cap_overall"])
                overall_100 = min(overall_100, cap)
            if "penalty" in act:
                overall_100 = max(0.0, overall_100 - float(act["penalty"]))

            applied.append({
                "gate": rule.name,
                "criterion": crit,
                "criterion_score": val,
                "before_100": before,
                "after_100": overall_100,
                "action": act,
                "note": rule.note,
            })

        overall_1_to_5_after = (overall_100 / 100.0) * 5.0
        return overall_1_to_5_after, applied

    def evaluate(
        self,
        scores_1_to_5: Dict[str, float],
        stage: str,
        industry: Optional[str] = None,
        business_model: Optional[str] = None,
        override_stage_weights: Optional[Dict[str, float]] = None,
        extra_multipliers: Optional[Dict[str, float]] = None,
        apply_gating: bool = True,
    ) -> Dict[str, Any]:
        """
        한 번에:
        - 최종 weights 산출
        - 가중 점수(1~5)
        - gate 적용(선택)
        """
        weights = self.compute_weights(
            stage=stage,
            industry=industry,
            business_model=business_model,
            override_stage_weights=override_stage_weights,
            extra_multipliers=extra_multipliers,
        )
        overall = self.score(scores_1_to_5, weights)

        gate_logs: List[Dict[str, Any]] = []
        overall_after = overall
        if apply_gating:
            overall_after, gate_logs = self.apply_gates(
                stage=stage,
                industry=industry,
                business_model=business_model,
                score_by_criterion=scores_1_to_5,
                overall_1_to_5=overall,
            )

        return {
            "stage": self.normalize_stage(stage),
            "industry": industry,
            "business_model": business_model,
            "weights": weights,                   # 합 100
            "overall_1_to_5_raw": round(overall, 4),
            "overall_1_to_5_after_gates": round(overall_after, 4),
            "overall_100_raw": round((overall / 5.0) * 100.0, 2),
            "overall_100_after_gates": round((overall_after / 5.0) * 100.0, 2),
            "gates_applied": gate_logs,
        }


# -----------------------------
# 6) 사용 예시(로컬 테스트)
# -----------------------------
if __name__ == "__main__":
    engine = WeightEngine()

    # 예: Series B + B2B SaaS + subscription
    sample_scores = {
        "problem_definition": 3.5,
        "solution_product": 3.8,
        "market": 3.6,
        "business_model": 3.4,
        "competitive_advantage": 3.2,
        "growth_strategy": 3.1,
        "team": 3.7,
        "financial_plan": 2.8,   # Series B+에서 낮으면 gate penalty 가능
        "risk_management": 3.0,
    }

    out = engine.evaluate(
        scores_1_to_5=sample_scores,
        stage="Series B",
        industry="b2b_saas",
        business_model="subscription_saas",
        apply_gating=True,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))