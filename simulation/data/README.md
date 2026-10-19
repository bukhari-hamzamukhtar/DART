# Spring training data

The exact frames the spring network (`jet_brain_v3.pt`) was trained on: one file per
curriculum stage, written by `simulation/headless_sim3.py`. That generator is not
seeded, so running it again produces a different set; these files are the original.

They are gzipped to stay small (161 MB of JSONL, 29 MB here). To use them where the
training code expects plain `.jsonl`:

```bash
gunzip -k simulation/data/*.jsonl.gz
```

| file | frames (lines) |
|---|---|
| `data_slow_singles.jsonl.gz` | 94,194 |
| `data_fast_homing.jsonl.gz` | 73,040 |
| `data_multi_threat.jsonl.gz` | 65,312 |
| `data_cluster_swarm.jsonl.gz` | 54,751 |
| `data_adversarial.jsonl.gz` | 42,837 |
| `data_cross_attention.jsonl.gz` | 260,918 |
