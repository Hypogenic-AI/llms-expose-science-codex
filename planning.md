# Research Plan

## Research Question
Can large language models, when given bibliometric slices of real-world domains, surface under-researched problem areas more effectively when grounded in structured attention/impact signals than when prompted generically?

## Background and Motivation
AI for science often reinforces well-funded agendas rather than exposing neglected needs (e.g., women’s health, climate adaptation for vulnerable regions). We have two OpenAlex snapshots (climate adaptation; neglected tropical diseases) and survey-style papers on scientific LLMs and autonomous research agents. This project tests whether LLMs can act as gap detectors by contrasting grounded prompts (with citation/geography/OA statistics) against ungrounded prompts, and by comparing LLM outputs to simple numeric baselines.

## Hypothesis Decomposition
- H1: Providing structured attention indicators (citation density, open access share, Global South authorship share) leads LLMs to produce more grounded gap statements (more numeric references, region mentions) than generic prompts.
- H2: LLM-identified gaps will align better with quantitative under-attention signals (e.g., low Global South authorship, low OA) when grounded prompts are used than in zero-shot.
- Variables: independent = prompt condition (generic vs grounded). Dependent = grounding score (numeric references), region-mention score, gap-coverage score vs heuristic under-attention indicators. Control = same model (gpt-4.1 via OpenRouter), same data slices.
- Success: statistically significant increase in grounding/region/coverage scores for grounded prompts.

## Proposed Methodology

### Approach
Combine lightweight bibliometric analytics with LLM prompting. Compute attention proxies (citation counts, OA rate, author country distribution, top concepts) for both datasets. Run LLM gap-detection prompts in two conditions (generic vs grounded with stats) and score responses automatically against heuristic signals.

### Experimental Steps
1. Data validation: load both OpenAlex JSON files; sanity-check counts, missing fields; extract key attributes.
2. Attention metrics: compute per-dataset citation mean/median, OA share, Global South authorship share, top concepts/countries.
3. Heuristic under-attention flags: identify countries with low authorship counts relative to vulnerability proxy (Global South) and low-OA slices.
4. Prompt design: two prompts per dataset—(A) generic gap-detection using top titles/abstract snippets; (B) grounded prompt supplying computed stats + representative low-attention samples. Fix temperature/seed.
5. LLM execution: call gpt-4.1 via OpenRouter; store raw outputs with metadata (prompt, model, temperature, timestamp) in results/.
6. Scoring: regex-based metrics—numeric grounding count, region-mention count (Africa, South Asia, Latin America, LMIC/Global South), gap-category coverage (equity, funding, implementation, data scarcity, OA). Align with heuristic under-attention signals.
7. Statistical analysis: paired comparison of scores between prompt conditions per dataset (Wilcoxon signed-rank); report effect sizes and confidence via bootstrap.
8. Visualization: bar plots of scores and attention metrics.
9. Documentation: compile REPORT.md with methods, prompts, metrics, results, limitations; update README.md.

### Baselines
- Numeric heuristic baseline: direct attention metrics (e.g., Global South authorship share, OA share) without LLM.
- LLM zero-shot gap detection (generic prompt) as primary baseline.

### Evaluation Metrics
- Grounding score: count of numeric expressions in response referencing evidence.
- Region-mention score: count of mentions of under-represented regions/LMIC terms.
- Gap-coverage score: count of distinct gap categories hit (equity/geography, funding/resources, implementation/practice, data sparsity, OA/policy).
- Alignment score: overlap between gaps cited and heuristic under-attention signals (e.g., mentions of low Global South share when heuristic indicates low share).

### Statistical Analysis Plan
- Within-dataset paired comparison (generic vs grounded) using Wilcoxon signed-rank on scores (n=2 responses per condition * datasets; if insufficient, augment with multiple samples by re-prompting with different subsets of titles).
- Bootstrap 10k resamples for mean difference CI; report median and 95% CI.
- Significance level α=0.05; acknowledge small-n limitations.

## Expected Outcomes
Grounded prompts should increase grounding/region/coverage scores and better match heuristic under-attention signals. If not, it suggests LLMs default to generic equity tropes without evidence, challenging the hypothesis.

## Timeline and Milestones
- 0:10 Data inspection & metrics implementation
- 0:20 Prompt design and LLM runner
- 0:25 Experiments and scoring
- 0:15 Analysis + plots
- 0:20 Reporting (REPORT.md, README.md)

## Potential Challenges
- Small dataset size may limit statistical power; mitigate by sampling multiple prompt variants.
- Missing country codes in authorships; fallback to affiliation strings and concept geography proxies.
- API variability/cost; cap outputs and cache responses.
- Regex scoring may undercount nuanced gaps; supplement with manual spot-check in analysis.

## Success Criteria
- Reproducible pipeline with saved prompts/outputs and scores.
- Observed improvement in at least two metrics (grounding, region mentions, gap coverage) for grounded prompts vs baseline.
- Clear documentation of limitations and next steps.
