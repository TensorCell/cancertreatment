# Data

Not checked in. Two files are needed:

| file | content | source |
|---|---|---|
| `data_400k.csv` | 400 000 protocols x 21 rows (20 dose slots + target row) | Hugging Face `zzox531/emt6ro-sobol-400k` |
| `data_400k.csv.series_map.csv` | new series id -> (source, original id) | same repository |

```bash
huggingface-cli download zzox531/emt6ro-sobol-400k \
    data_400k.csv data_400k.csv.series_map.csv SHA256SUMS_400k \
    --repo-type dataset --local-dir data
(cd data && sha256sum -c SHA256SUMS_400k)
```

The original uniform 200k set alone (`data.csv`) is embedded in `data_400k.csv` and can
also be rebuilt from the paper authors' repository:

```bash
python -m cancer_nn.preprocess_data
```

Columns: `time` (simulation step of the dose, 6 s per step, 0 to 72 000 = first 5 days),
`dose` (Gy, multiple of 0.125), `time_gap`, `series`, `time_idx`, `is_target`, `target`
(mean live tumour cells at step 144 000 = day 10, over 100 replicates).

The 400k test split is half Sobol protocols with a wider dose range, so its raw test
metrics are not comparable with the 200k split; filter to `source == uniform` through
the series map for a like-for-like comparison.
