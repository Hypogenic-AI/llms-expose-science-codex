# REPORT: Can LLMs Expose What Science Refuses to See?

## 1. Executive Summary
- Research question: can LLM prompts grounded in bibliometric attention signals surface neglected scientific gaps better than generic prompts?
- Key finding: with small OpenAlex slices, grounded prompts using a local Qwen2.5-1.5B model did **not** improve numeric grounding and only slightly shifted region/gap mentions; quantitative baselines already revealed strong attention imbalance (e.g., Global South authorship share 13.5% in climate adaptation, 30% in NTD).
- Implication: structured stats alone were insufficient with a lightweight model; stronger models or richer impact signals are needed to make LLMs reliable gap detectors.

## 2. Goal
- Hypothesis: structured attention/impact indicators (citations, OA rate, Global South authorship share) will lead LLMs to produce more grounded and aligned gap statements than generic prompts.
- Importance: tests whether LLMs can critique scientific attention patterns instead of reinforcing dominant agendas.
- Expected impact: a template for automated "gap audits" that combine bibliometrics and LLM reasoning.

## 3. Data Construction

### Dataset Description
- `OpenAlex Climate Adaptation` (200 works, CC0): query `search=climate adaptation`.
- `OpenAlex Neglected Tropical Diseases` (200 works, CC0): query `search=neglected tropical disease`.
- Fields used: titles, publication year, citation count, open_access flags, concepts, authorship institution country codes.

### Example Samples
- Climate: Managing Climate Change Refugia for Climate Adaptation (2016, cites=499, OA=True, countries=US); Roadmap towards justice in urban climate adaptation research (2016, cites=544, OA=False, countries=CA,ES,GB,NL,US,ZA).
- NTD: Control of Neglected Tropical Diseases (2007, cites=1538, OA=False, countries=CH,GB,US); Helminth infections: the great neglected tropical diseases (2008, cites=1540, OA=True, countries=US).

### Data Quality
- Missing country codes for some highly cited works; otherwise citation/OA fields present for all 200 entries per set.
- No duplicates detected; JSON structure consistent with OpenAlex schema.
- Class distribution: all single-slice corpora (no labels); Global South authorship share low (13.5% climate, 30% NTD).

### Preprocessing Steps
1. Load JSON (`results` field) → pandas DataFrame.
2. Extract authorship institution country codes; flag `has_global_south` via a World Bank-like list.
3. Compute OA flag, citation stats, top concepts/countries.
4. Sample low-citation works (preferring Global South authorships) for prompts.

### Train/Val/Test Splits
- Not applicable; evaluation is prompt-based on full slices. Random seed=42 governs sampling order.

## 4. Experiment Description

### Methodology
- Compare two prompting conditions per dataset:
  - **Generic**: list 5 representative works (title/year/citations/countries/concepts); ask for 3–5 gaps.
  - **Grounded**: provide attention indicators (mean/median citations, OA rate, Global South authorship share, top countries/concepts, heuristic flags) plus the same examples; ask to reference numbers.
- LLM: `Qwen2.5-1.5B-Instruct` via `transformers` (fallback used because no OPENROUTER_API_KEY was available to call GPT-4.1/OpenRouter).
- Scoring (regex-based): numeric grounding count; region mentions (Africa/Latin America/Global South/LMIC etc.); gap-category coverage (equity/funding/implementation/data/OA); alignment hits vs heuristic flags (low GS, low OA, country concentration).

### Implementation Details
- Tools: Python 3.10.12; pandas 2.3.3; numpy 2.2.6; scipy 1.15.3; seaborn 0.13.2; transformers 4.57.3; torch 2.9.1; openai 2.11.0 (unused due to missing key).
- Scripts: `src/run_experiments.py` (data prep, prompting, scoring), `src/analyze_results.py` (plots, CSV summary).
- Outputs: `results/llm_outputs/llm_runs.json`, `results/metrics_summary.csv`, plots in `results/plots/`.

### Hyperparameters
| Parameter | Value | Selection Method |
|-----------|-------|------------------|
| seed | 42 | fixed for reproducibility |
| temperature | 0.2 | low-variance generation |
| max_new_tokens | 120 | cap runtime, ensure multi-bullet outputs |
| examples per prompt | 5 | balance context length/runtime |

### Analysis Pipeline
1. Compute attention metrics (citations, OA rate, Global South share, top countries/concepts).
2. Generate prompts (generic vs grounded) with sampled examples.
3. Run LLM once per condition × dataset; save raw responses.
4. Score responses with regex-based metrics; compute paired differences and bootstrap CIs (n=1 pair → descriptive only).
5. Visualize LLM scores and attention proxies.

### Reproducibility
- Hardware: CPU (no GPU); model loaded locally (~1.5B parameters).
- Runs per condition: 1 (cost/runtime constrained by local model fallback).
- Seeds: 42 for sampling; deterministic scoring.

## 5. Results

### Attention Baselines (heuristics)
- Climate adaptation: n=200; mean citations 810.3, median 252; OA rate 42.5%; Global South authorship share 13.5%; top country share US 36% → strong North-heavy attention.
- Neglected tropical diseases: n=200; mean citations 170.3, median 97.5; OA rate 70.5%; Global South authorship share 30%; US share 48% → better OA but still concentrated.

### LLM Output Metrics
| Dataset | Condition | Grounding count | Region mentions | Gap coverage | Alignment hits |
|---------|-----------|-----------------|-----------------|--------------|----------------|
| Climate | Generic | 0 | 3 | 1 | 2 |
| Climate | Grounded | 0 | 2 | 2 | 2 |
| NTD | Generic | 0 | 2 | 2 | 2 |
| NTD | Grounded | 0 | 1 | 2 | 2 |

- No numeric grounding appeared in any output (grounding_count=0 throughout).
- Region mentions increased slightly for grounded NTD in the first attempt but decreased after rerun; small-n noise dominates.
- Gap-category coverage improved for climate when grounded (1 → 2 categories) but stayed flat for NTD.
- Alignment hits (mentions matching heuristic flags) unchanged across conditions.

### Visualizations
- `results/plots/attention_proxies.png`: bar chart of OA rate vs Global South share per dataset (shows stark OA gap in climate and low GS share across both).
- `results/plots/llm_scores.png`: bar chart of region/gap/alignment scores by condition (shows marginal differences, all low).

## 6. Analysis
- **Hypothesis test**: Grounded prompts did not increase numeric grounding and only modestly affected region/gap mentions; with n=1 per condition the Wilcoxon/bootstrap stats are descriptive (all p≈1). Evidence does **not** support the hypothesis in this configuration.
- **Baseline vs LLM**: Simple heuristics already expose attention gaps (GS share 13.5% climate, 30% NTD; OA 42.5% vs 70.5%). The lightweight LLM failed to leverage provided numbers, suggesting current setup adds little beyond heuristics.
- **Error patterns**: Outputs truncated mid-bullet due to token cap; no percentages copied despite explicit instructions; generic prompts sometimes mentioned regions more than grounded ones, likely due to model prior rather than data.
- **Limitations**: Missing API key prevented use of GPT-4.1/OpenRouter; small local model (1.5B) and single-sample runs limit power; scoring via regex undercounts nuanced grounding; datasets are small (200 works) and lack impact signals (funding, burden).

## 7. Conclusions
- Baseline bibliometrics already reveal significant attention asymmetry: climate adaptation is highly cited yet poorly open-access and heavily skewed to high-income authors; NTDs have better OA but remain concentrated.
- With a small local model, grounding prompts in attention metrics did **not** yield stronger or more evidence-based gap statements; numeric grounding remained zero.
- The hypothesis that structured signals improve LLM gap detection is unsupported under these constraints; stronger models and richer evidence are required.

## 8. Next Steps
1. Re-run with GPT-4.1/Claude via OpenRouter once credentials are available; increase runs and vary temperature to get variance estimates.
2. Enrich attention/impact signals (e.g., funder data, disease burden, vulnerability indices) and include them in grounded prompts to test calibration.
3. Expand scoring to include factual consistency (LLM claims vs provided stats) and human evaluation of usefulness.
4. Increase sample size per condition (multiple prompt variants, different title subsets) and add numeric grounding coaching (few-shot examples using provided stats).
5. Explore lightweight finetuning or RAG that injects citation/OA stats directly into the answer to encourage numeric references.
