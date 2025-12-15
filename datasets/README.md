# Downloaded Datasets

This directory stores lightweight bibliometric snapshots for probing gaps between attention and impact. Bulk data are excluded from git; see `.gitignore`.

## Dataset 1: OpenAlex – Climate Adaptation Works
- **Source**: `https://api.openalex.org/works?search=climate%20adaptation&per_page=200`
- **Size**: 200 records (JSON)
- **Format**: JSON list of works with concepts, citations, affiliations, and open-access metadata
- **Task**: Gap analysis for climate adaptation research attention vs. impact
- **Splits**: Single JSON response
- **License**: OpenAlex API responses are released under CC0

### Download Instructions
- Refresh full snapshot:
```bash
curl -L 'https://api.openalex.org/works?search=climate%20adaptation&per_page=200' \
  -o datasets/openalex_climate_adaptation.json
```
- Paginate for larger pulls (replace `cursor=*` to traverse):
```bash
curl -L 'https://api.openalex.org/works?search=climate%20adaptation&per_page=200&cursor=*' \
  -o datasets/openalex_climate_adaptation_page1.json
```

### Loading Example
```python
import json
data = json.load(open("datasets/openalex_climate_adaptation.json"))
works = data["results"]
print(len(works), works[0]["display_name"])
```

### Sample Data
See `datasets/openalex_climate_adaptation_samples.json` (first 5 works).

## Dataset 2: OpenAlex – Neglected Tropical Disease Works
- **Source**: `https://api.openalex.org/works?search=neglected%20tropical%20disease&per_page=200`
- **Size**: 200 records (JSON)
- **Format**: JSON list of works with metadata and citations
- **Task**: Contrast attention on neglected diseases against better-funded topics
- **Splits**: Single JSON response
- **License**: OpenAlex API responses are released under CC0

### Download Instructions
```bash
curl -L 'https://api.openalex.org/works?search=neglected%20tropical%20disease&per_page=200' \
  -o datasets/openalex_neglected_tropical_disease.json
```
- Paginate for more coverage as above with `cursor=*`.

### Loading Example
```python
import json
data = json.load(open("datasets/openalex_neglected_tropical_disease.json"))
works = data["results"]
```

### Sample Data
See `datasets/openalex_neglected_tropical_disease_samples.json`.

## Notes
- Both datasets are small, API-derived, and CC0; safe to expand via pagination for deeper analyses.
- When computing “attention vs. need”, combine citation counts, fields/concepts, OA status, and author country to proxy funding or equity gaps.
