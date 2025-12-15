## Resources Catalog

### Summary
Collected six papers, two lightweight OpenAlex datasets, and two code repositories to study how LLMs can expose under-researched scientific problems.

### Papers
Total papers downloaded: 6

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| A Comprehensive Survey of Scientific LLMs and Their Applications in Scientific Discovery | Yu Zhang et al. | 2024 | `papers/2406.10833v3_scientific_llms_survey.pdf` | Survey of 260+ scientific LLMs, benchmarks, and datasets |
| Automated Literature Review Using NLP Techniques and LLM-Based RAG | Nurshat Fateh Ali et al. | 2024 | `papers/2411.18583v1_automated_lit_review_rag.pdf` | Compares spaCy, T5, GPT-3.5 RAG on SciTLDR |
| AI Literature Review Suite | David A. Tovar | 2023 | `papers/2308.02443v1_ai_lit_review_suite.pdf` | End-to-end LLM-powered literature review toolkit |
| Learning From Failure: Integrating Negative Examples when Fine-tuning LLMs as Agents | Renxi Wang et al. | 2024 | `papers/2402.11651v2_learning_from_failure_agent_tuning.pdf` | Agent tuning with failed trajectories improves robustness |
| Jr. AI Scientist and Its Risk Report | Atsuyuki Miyai et al. | 2025 | `papers/2511.04583v2_jr_ai_scientist.pdf` | Autonomous research workflow plus risk assessment |
| Exploring the use of AI authors and reviewers at Agents4Science | Federico Bianchi et al. | 2025 | `papers/2511.15534v1_ai_authors_agents4science.pdf` | Case study of AI agents as authors and reviewers |

See `papers/README.md` for details.

### Datasets
Total datasets downloaded: 2

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| OpenAlex – Climate Adaptation Works | OpenAlex API search | 200 works | Bibliometric gap analysis | `datasets/openalex_climate_adaptation.json` | CC0; sample in `_samples.json` |
| OpenAlex – Neglected Tropical Disease Works | OpenAlex API search | 200 works | Bibliometric gap analysis | `datasets/openalex_neglected_tropical_disease.json` | CC0; sample in `_samples.json` |

See `datasets/README.md` for download instructions.

### Code Repositories
Total repositories cloned: 2

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| jr-ai-scientist | https://github.com/Agent4Science-UTokyo/Jr.AI-Scientist | Autonomous AI scientist workflow and risk report | `code/jr-ai-scientist` | Code release pending; docs and figures available |
| awesome-scientific-llms | https://github.com/yuzhimanhua/Awesome-Scientific-Language-Models | Curated catalog of scientific LLMs | `code/awesome-scientific-llms` | Good for finding domain-specific checkpoints/benchmarks |

See `code/README.md` for details.

### Resource Gathering Notes
- **Search Strategy**: Queried arXiv via API for “scientific LLMs”, “AI scientist”, and automated literature review. Selected recent (2023–2025) works with agent workflows or literature-mining focus. Used OpenAlex API to pull bibliometric slices for climate adaptation and neglected tropical diseases.
- **Selection Criteria**: Prioritized open-access papers with agentic or survey contributions, code availability, and relevance to identifying neglected research areas. Chose CC0 bibliometric data to avoid licensing issues.
- **Challenges Encountered**: `wget` unavailable due to missing OpenSSL libs; switched to `curl`. OpenAlex `topic.display_name` filter invalid; replaced with `search` queries.
- **Gaps and Workarounds**: Funding-level datasets not included due to size/access; recommend augmenting OpenAlex pulls with SDG tags or grant databases if needed.

### Recommendations for Experiment Design
1. **Primary dataset(s)**: Expand OpenAlex slices via `cursor=*` for target domains (e.g., climate adaptation, neglected diseases, environmental justice) to compute attention vs. impact metrics.
2. **Baseline methods**: Compare keyword/TF-IDF summarization vs. RAG-LLMs for gap reports; run agent baselines with and without failure-trajectory tuning.
3. **Evaluation metrics**: ROUGE for summaries; citation density per concept/year; OA/share of Global South authorships; agent task completion and hallucination/error rates.
4. **Code to adapt/reuse**: Use jr-ai-scientist workflow for agent orchestration and risk checks; mine awesome-scientific-llms to locate domain-specific checkpoints and benchmarks.
