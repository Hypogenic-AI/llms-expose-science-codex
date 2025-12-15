## Literature Review

### Research Area Overview
The project examines how LLMs can surface under-studied scientific problems by automating literature synthesis, benchmarking attention vs. impact, and running autonomous research cycles. Recent work spans surveys of scientific LLMs, automated literature-review pipelines, agent fine-tuning for robustness, and experiential reports of AI co-scientists.

### Key Papers

#### A Comprehensive Survey of Scientific Large Language Models and Their Applications in Scientific Discovery
- **Authors**: Yu Zhang et al.
- **Year**: 2024 (arXiv:2406.10833)
- **Source**: arXiv preprint
- **Key Contribution**: Catalogs 260+ scientific LLMs across modalities; maps pre-training data, instruction tuning, and evaluation tasks.
- **Methodology**: Taxonomy of encoder, decoder, and encoder-decoder pretraining (MLM, next-token prediction, instruction tuning); cross-field comparison.
- **Datasets Used**: Aggregates domain datasets (papers, proteins, molecules, tables, climate time series) but no new dataset collection.
- **Results**: Highlights cross-modal transfer patterns and gaps (e.g., limited benchmarks for geoscience and materials).
- **Code Available**: Yes, curated list (awesome-scientific-llms repo).
- **Relevance to Our Research**: Provides landscape view and points to domains with sparse benchmarks—useful for selecting neglected areas.

#### Automated Literature Review Using NLP Techniques and LLM-Based Retrieval-Augmented Generation
- **Authors**: Nurshat Fateh Ali et al.
- **Year**: 2024 (arXiv:2411.18583)
- **Source**: arXiv preprint
- **Key Contribution**: Compares spaCy keyword baseline, SimpleT5 summarizer, and GPT-3.5 RAG for auto literature review.
- **Methodology**: Pipeline from PDFs → text → summarization; evaluated on SciTLDR using ROUGE-1/2/L.
- **Datasets Used**: SciTLDR (scientific paper summarization).
- **Results**: GPT-3.5 RAG tops with ROUGE-1 ≈ 0.36; transformer and frequency approaches lag.
- **Code Available**: Not linked; straightforward to reimplement with open models.
- **Relevance to Our Research**: Demonstrates gains from RAG; suggests using citation-aware retrieval to reduce hallucinations when surfacing gaps.

#### AI Literature Review Suite
- **Author**: David A. Tovar
- **Year**: 2023 (arXiv:2308.02443)
- **Source**: arXiv preprint
- **Key Contribution**: End-to-end suite for searching, downloading, organizing, and summarizing papers with embeddings + LLMs.
- **Methodology**: Semantic search, PDF ingestion, GUI for querying and summarizing with LLMs; bibliographic organization.
- **Datasets Used**: Uses open-access papers; no fixed benchmark.
- **Results**: Qualitative demonstration of workflow efficiency.
- **Code Available**: Implied but not provided in paper.
- **Relevance to Our Research**: Blueprint for building an agentic literature-mining tool to discover neglected topics.

#### Learning From Failure: Integrating Negative Examples when Fine-tuning Large Language Models as Agents
- **Authors**: Renxi Wang et al.
- **Year**: 2024 (arXiv:2402.11651)
- **Source**: arXiv preprint
- **Key Contribution**: Shows failed tool-use trajectories improve agent performance when kept with success labels.
- **Methodology**: Collect tool-use traces; prefix/suffix cues for success vs. failure; fine-tune open models; evaluate on math, multi-hop QA, strategic QA.
- **Datasets Used**: Agentic benchmarks (math, QA tasks).
- **Results**: Meaningful accuracy gains vs. success-only fine-tuning; better robustness to noisy trajectories.
- **Code Available**: Not linked in PDF; approach is simple to replicate.
- **Relevance to Our Research**: Encourages keeping “false starts” when training LLMs to identify and iterate on neglected problem statements.

#### Jr. AI Scientist and Its Risk Report: Autonomous Scientific Exploration from a Baseline Paper
- **Authors**: Atsuyuki Miyai et al.
- **Year**: 2025 (arXiv:2511.04583)
- **Source**: arXiv preprint
- **Key Contribution**: Autonomous system that reads a baseline paper, proposes hypotheses, runs experiments, and drafts improved papers; provides a risk report.
- **Methodology**: Agent workflow with planning, coding agents, experiment execution, automated reviews.
- **Datasets Used**: Builds on public ML baselines (e.g., LoCoOp, Min-K%++, GL-MCM).
- **Results**: Generated papers evaluated by AI and human reviewers; highlights risks like review-score gaming and fabricated results.
- **Code Available**: Repo cloned (documentation + figures; core code withheld).
- **Relevance to Our Research**: Offers evaluation axes and risk controls for autonomous “gap-finding” scientists.

#### Exploring the use of AI authors and reviewers at Agents4Science
- **Authors**: Federico Bianchi et al.
- **Year**: 2025 (arXiv:2511.15534)
- **Source**: arXiv preprint
- **Key Contribution**: Case study of a venue where AI agents acted as primary authors and reviewers.
- **Methodology**: Organizes submissions and reviews with AI agents; reports observations on creativity, rigor, and collaboration patterns.
- **Datasets Used**: Conference submissions; not public.
- **Results**: AI agents can produce viable drafts but need human oversight for correctness and novelty checks.
- **Code Available**: Not linked.
- **Relevance to Our Research**: Provides socio-technical insights on using AI to surface and vet research ideas.

### Common Methodologies
- Retrieval-augmented generation for grounded summarization (papers 2, 3).
- Agent fine-tuning and trajectory learning (paper 4) to improve tool use and planning.
- Autonomous research workflows with iterative experimentation (paper 5).
- Large curated model/dataset taxonomies to spot sparse coverage areas (paper 1).

### Standard Baselines
- Summarization: ROUGE on SciTLDR; spaCy keyword vs. transformer vs. GPT-style LLMs.
- Agent tasks: math/QA benchmarks for tool-use success rates.
- Research agents: human/AI review scores of generated papers; reproducibility checks.

### Evaluation Metrics
- ROUGE-1/2/L for literature-summary quality.
- Success rate / completion rate of agent tool-use trajectories.
- Citation counts, OA status, and field coverage (from bibliometric APIs) as proxy metrics for attention vs. impact gaps.
- Human/LLM review scores for AI-generated research artifacts.

### Datasets in the Literature
- SciTLDR for summarization.
- Domain-specific scientific corpora (papers, proteins, molecules) cataloged in the survey.
- Agent benchmarks for math and multi-hop QA (tool-use tuning).
- Our added OpenAlex slices (climate adaptation; neglected tropical diseases) to probe attention gaps.

### Gaps and Opportunities
- Limited benchmarks explicitly measuring “neglect” or mismatch between societal impact and research attention.
- Scarce open datasets on funding flows vs. publication density; need API-derived proxies (OpenAlex concepts, SDG tags, country codes).
- Risk evaluation of AI scientists is nascent; reproducibility and fabrication detection remain weak.
- Few standardized metrics for idea novelty and coverage of underrepresented communities/topics.

### Recommendations for Our Experiment
- **Recommended datasets**: OpenAlex slices for climate adaptation and neglected tropical diseases; expand via `cursor=*` for larger samples. Consider adding SDG-tag filters to contrast attention across goals.
- **Recommended baselines**: spaCy/keyword and transformer summarizers vs. RAG-LLMs for literature gap reports; agent success-rate baselines without failure trajectories.
- **Recommended metrics**: ROUGE for summaries; citation density per concept/year; OA rate; geographic diversity of authorships; agent task completion and hallucination/error rate during hypothesis generation.
- **Methodological considerations**: Preserve failed agent trajectories for fine-tuning; use RAG with citation-aware retrieval to ground claims; add human-in-the-loop verification for suggested neglected topics; track novelty via overlap with existing concepts and funding data when available.
