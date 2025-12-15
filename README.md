# LLMs as Gap Detectors

Brief experiments on whether LLMs can expose neglected scientific problems by grounding prompts in bibliometric attention signals.

## Key Findings
- Baseline bibliometrics already show attention skew: climate adaptation has low OA (42.5%) and only 13.5% Global South authorship; NTDs have higher OA (70.5%) but 48% of works come from the US.
- Grounded prompts using a local Qwen2.5-1.5B model did **not** increase numeric grounding (all zero) and only small, inconsistent shifts in region/gap mentions.
- Small model fallback (no OpenRouter key available) and token caps limited output richness; results are descriptive, not statistically strong.

## Reproduction
1. `uv venv --python python3.10 && source .venv/bin/activate`
2. `uv sync`
3. Run experiments: `python src/run_experiments.py`
4. Generate plots/CSV: `python src/analyze_results.py`

Outputs: `results/llm_outputs/llm_runs.json`, `results/metrics_summary.csv`, plots in `results/plots/`.

## File Structure
- `planning.md` – research plan
- `src/run_experiments.py` – data prep, prompts, scoring
- `src/analyze_results.py` – plots and metrics table
- `datasets/` – OpenAlex slices (local, CC0)
- `results/` – saved runs, metrics, and plots
- `REPORT.md` – full report with methodology and results
