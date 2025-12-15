import json
import os
import random
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from openai import OpenAI
from scipy import stats
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Countries roughly aligned with World Bank low/middle income regions to proxy Global South.
GLOBAL_SOUTH_CODES = {
    "AF",
    "AL",
    "AO",
    "AR",
    "AZ",
    "BD",
    "BF",
    "BJ",
    "BO",
    "BR",
    "BW",
    "CI",
    "CL",
    "CM",
    "CO",
    "CR",
    "CU",
    "DJ",
    "DZ",
    "EC",
    "EG",
    "ET",
    "GH",
    "GT",
    "HN",
    "HT",
    "ID",
    "IN",
    "JO",
    "KE",
    "KG",
    "KH",
    "KR",
    "KZ",
    "LA",
    "LB",
    "LK",
    "MA",
    "MD",
    "ME",
    "MG",
    "ML",
    "MM",
    "MN",
    "MR",
    "MX",
    "MZ",
    "NA",
    "NG",
    "NI",
    "NP",
    "PA",
    "PE",
    "PH",
    "PK",
    "PS",
    "PY",
    "RS",
    "RU",
    "RW",
    "SA",
    "SD",
    "SN",
    "SO",
    "SY",
    "TH",
    "TJ",
    "TN",
    "TR",
    "TZ",
    "UA",
    "UG",
    "UY",
    "UZ",
    "VE",
    "VN",
    "ZA",
    "ZM",
    "ZW",
}

# Keywords grouped into coarse gap categories for automated scoring.
GAP_CATEGORY_TERMS = {
    "equity_geography": ["equity", "inequity", "under-represent", "global south", "africa", "latin america", "south asia", "southeast asia", "low-income", "lmic"],
    "funding_capacity": ["funding", "investment", "resources", "capacity", "grant"],
    "implementation": ["implementation", "deployment", "uptake", "service", "clinic", "field", "local system", "health system"],
    "data_scarcity": ["data", "surveillance", "reporting", "dataset", "measurement"],
    "open_access": ["open access", "paywall", "license", "oa"],
}

SYSTEM_PROMPT = "You are a policy-savvy research auditor who surfaces neglected scientific gaps using provided evidence only."


@dataclass
class AttentionMetrics:
    dataset: str
    n_works: int
    mean_citations: float
    median_citations: float
    oa_rate: float
    gs_share: float
    top_countries: List[Tuple[str, int]]
    top_concepts: List[Tuple[str, int]]


@dataclass
class LLMScore:
    grounding_count: int
    region_mentions: int
    gap_coverage: int
    alignment_hits: int


@dataclass
class LLMRun:
    dataset: str
    condition: str
    prompt: str
    response: str
    scores: LLMScore
    stats: AttentionMetrics
    timestamp: str
    model: str
    temperature: float


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_works(path: Path) -> List[dict]:
    data = json.load(open(path))
    return data["results"]


def parse_countries(authorships: List[dict]) -> List[str]:
    codes = set()
    for a in authorships or []:
        for inst in a.get("institutions") or []:
            code = inst.get("country_code")
            if code:
                codes.add(code)
    return sorted(codes)


def parse_concepts(concepts: List[dict], top_k: int = 8) -> List[str]:
    sorted_concepts = sorted(concepts or [], key=lambda c: c.get("score", 0), reverse=True)
    names = [c.get("display_name") for c in sorted_concepts if c.get("display_name")]
    return names[:top_k]


def works_to_dataframe(works: List[dict]) -> pd.DataFrame:
    rows = []
    for w in works:
        countries = parse_countries(w.get("authorships"))
        concepts = parse_concepts(w.get("concepts"))
        rows.append(
            {
                "id": w.get("id"),
                "title": w.get("display_name") or w.get("title") or "",
                "year": w.get("publication_year"),
                "cited_by": w.get("cited_by_count", 0) or 0,
                "is_oa": bool((w.get("open_access") or {}).get("is_oa", False)),
                "countries": countries,
                "has_global_south": any(c in GLOBAL_SOUTH_CODES for c in countries),
                "concepts": concepts,
            }
        )
    df = pd.DataFrame(rows)
    return df


def compute_attention_metrics(dataset: str, df: pd.DataFrame) -> AttentionMetrics:
    country_counts = (
        df["countries"]
        .explode()
        .dropna()
        .value_counts()
        .head(8)
        .reset_index()
        .values.tolist()
    )
    concept_counts = (
        df["concepts"]
        .explode()
        .dropna()
        .value_counts()
        .head(10)
        .reset_index()
        .values.tolist()
    )
    return AttentionMetrics(
        dataset=dataset,
        n_works=len(df),
        mean_citations=float(df["cited_by"].mean()),
        median_citations=float(df["cited_by"].median()),
        oa_rate=float(df["is_oa"].mean()),
        gs_share=float(df["has_global_south"].mean()),
        top_countries=[(str(a), int(b)) for a, b in country_counts],
        top_concepts=[(str(a), int(b)) for a, b in concept_counts],
    )


def select_examples(df: pd.DataFrame, k: int = 6) -> List[dict]:
    # Prefer low-citation works that include Global South authors; fall back to others.
    gs = df[df["has_global_south"]].sort_values("cited_by").head(k).to_dict("records")
    remainder = df[~df["has_global_south"]].sort_values("cited_by").head(k).to_dict("records")
    combined = (gs + remainder)[:k]
    if not combined:
        subset = df.sort_values("cited_by").head(k)
    else:
        subset = pd.DataFrame(combined)
    examples = []
    for _, row in subset.iterrows():
        examples.append(
            {
                "title": row["title"],
                "year": int(row["year"]) if not pd.isna(row["year"]) else None,
                "cited_by": int(row["cited_by"]),
                "is_oa": bool(row["is_oa"]),
                "countries": row["countries"],
                "concepts": row["concepts"][:5],
            }
        )
    return examples


def generic_prompt(dataset: str, examples: List[dict]) -> str:
    lines = [f"Dataset: {dataset}. Below are representative works (title | year | citations | countries | concepts)."]
    for i, ex in enumerate(examples, 1):
        lines.append(
            f"{i}. {ex['title']} | {ex['year']} | cites={ex['cited_by']} | countries={','.join(ex['countries']) or 'NA'} | concepts={', '.join(ex['concepts'])}"
        )
    lines.append(
        "Task: list 3-5 neglected research gaps or misalignments between real-world need and current attention. "
        "Use only the provided context. Return bullet points in the format 'gap: ...; evidence: ...'."
    )
    return "\n".join(lines)


def grounded_prompt(dataset: str, metrics: AttentionMetrics, examples: List[dict], heuristics: Dict[str, float]) -> str:
    lines = [
        f"Dataset: {dataset}. Use the structured indicators to identify neglected research gaps.",
        f"n_works={metrics.n_works}, mean_citations={metrics.mean_citations:.1f}, median_citations={metrics.median_citations:.1f}",
        f"oa_rate={metrics.oa_rate:.2f}, global_south_authorship_share={metrics.gs_share:.2f}",
        f"top_countries={metrics.top_countries}",
        f"top_concepts={metrics.top_concepts}",
        f"heuristic_flags={heuristics}",
        "Low-attention examples (title | year | citations | OA | countries | concepts):",
    ]
    for i, ex in enumerate(examples, 1):
        lines.append(
            f"{i}. {ex['title']} | {ex['year']} | cites={ex['cited_by']} | OA={ex['is_oa']} | countries={','.join(ex['countries']) or 'NA'} | concepts={', '.join(ex['concepts'])}"
        )
    lines.append(
        "Task: list 3-5 neglected research gaps. Explicitly reference the numeric indicators when making claims. "
        "Return bullet points formatted as 'gap: ...; evidence: ...'."
    )
    return "\n".join(lines)


def call_llm(prompt: str, model: str = "gpt-4.1", temperature: float = 0.2) -> Tuple[str, str]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=600,
        )
        return resp.choices[0].message.content.strip(), model

    # Fallback to a local open-source model when API keys are unavailable.
    generator = get_local_generator()
    chat_prompt = generator.tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    outputs = generator(
        chat_prompt,
        max_new_tokens=120,
        do_sample=True,
        temperature=temperature,
        return_full_text=False,
        eos_token_id=generator.tokenizer.eos_token_id,
        pad_token_id=generator.tokenizer.pad_token_id,
    )
    generated = outputs[0]["generated_text"]
    return generated.strip(), "Qwen2.5-1.5B-Instruct"


@lru_cache(maxsize=1)
def get_local_generator(model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )
    pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
    pipe.tokenizer.padding_side = "left"
    return pipe


def score_response(text: str, heuristics: Dict[str, float]) -> LLMScore:
    lower = text.lower()
    grounding = len(re.findall(r"\\d+(?:\\.\\d+)?%?", text))
    region_terms = ["africa", "latin america", "south asia", "southeast asia", "global south", "lmic", "low-income", "middle-income"]
    region_mentions = sum(lower.count(term) for term in region_terms)
    gap_hits = 0
    for terms in GAP_CATEGORY_TERMS.values():
        if any(t in lower for t in terms):
            gap_hits += 1

    alignment_hits = 0
    if heuristics.get("low_gs") and any(term in lower for term in ["global south", "lmic", "africa", "latin america", "south asia"]):
        alignment_hits += 1
    if heuristics.get("low_oa") and "open access" in lower:
        alignment_hits += 1
    if heuristics.get("concentrated_countries") and any(cc.lower() in lower for cc, _ in heuristics.get("top_countries", [])):
        alignment_hits += 1
    return LLMScore(
        grounding_count=grounding,
        region_mentions=region_mentions,
        gap_coverage=gap_hits,
        alignment_hits=alignment_hits,
    )


def compute_heuristics(metrics: AttentionMetrics) -> Dict[str, float]:
    concentrated = metrics.top_countries[0][1] / metrics.n_works > 0.35 if metrics.top_countries else False
    return {
        "low_gs": metrics.gs_share < 0.35,
        "low_oa": metrics.oa_rate < 0.6,
        "concentrated_countries": concentrated,
        "top_countries": metrics.top_countries,
    }


def paired_statistics(scores_a: List[int], scores_b: List[int]) -> Dict[str, float]:
    diff = np.array(scores_b) - np.array(scores_a)
    if len(diff) < 2 or np.all(diff == 0):
        p = 1.0
    else:
        _, p = stats.wilcoxon(scores_a, scores_b, zero_method="wilcox")
    boot = []
    for _ in range(5000):
        sample = np.random.choice(diff, size=len(diff), replace=True)
        boot.append(sample.mean())
    ci_lower, ci_upper = np.percentile(boot, [2.5, 97.5])
    return {"wilcoxon_p": float(p), "mean_diff": float(diff.mean()), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper)}


def run_experiments() -> Dict[str, dict]:
    set_seed(42)
    results_dir = Path("results/llm_outputs")
    results_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "OpenAlex Climate Adaptation": Path("datasets/openalex_climate_adaptation.json"),
        "OpenAlex Neglected Tropical Diseases": Path("datasets/openalex_neglected_tropical_disease.json"),
    }

    all_runs: List[LLMRun] = []
    for dataset_name, path in datasets.items():
        works = load_works(path)
        df = works_to_dataframe(works)
        metrics = compute_attention_metrics(dataset_name, df)
        heuristics = compute_heuristics(metrics)
        examples = select_examples(df, k=5)

        prompts = {
            "generic": generic_prompt(dataset_name, examples),
            "grounded": grounded_prompt(dataset_name, metrics, examples, heuristics),
        }
        for condition, prompt in prompts.items():
            response, used_model = call_llm(prompt)
            scores = score_response(response, heuristics)
            run = LLMRun(
                dataset=dataset_name,
                condition=condition,
                prompt=prompt,
                response=response,
                scores=scores,
                stats=metrics,
                timestamp=datetime.utcnow().isoformat(),
                model=used_model,
                temperature=0.2,
            )
            all_runs.append(run)

    # Aggregate statistics
    summary = {}
    for metric_name in ["grounding_count", "region_mentions", "gap_coverage", "alignment_hits"]:
        summary[metric_name] = {}
        for dataset in set(r.dataset for r in all_runs):
            a_scores = [getattr(r.scores, metric_name) for r in all_runs if r.dataset == dataset and r.condition == "generic"]
            b_scores = [getattr(r.scores, metric_name) for r in all_runs if r.dataset == dataset and r.condition == "grounded"]
            if a_scores and b_scores:
                summary[metric_name][dataset] = {
                    "generic": a_scores,
                    "grounded": b_scores,
                    "stats": paired_statistics(a_scores, b_scores),
                }

    output_json = {
        "runs": [asdict(r) for r in all_runs],
        "summary": summary,
    }
    with open(results_dir / "llm_runs.json", "w") as f:
        json.dump(output_json, f, indent=2)

    print("Saved LLM runs to results/llm_outputs/llm_runs.json")
    return output_json


if __name__ == "__main__":
    run_experiments()
