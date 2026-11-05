# Baseline: HF GPT-2 eager

GPU: Tesla T4

| batch | prompt_len | new_tokens | prefill_ms | generate_s | tokens_per_s | ms_per_step |
|---|---|---|---|---|---|---|
| 1 | 32 | 64 | 14.167 | 0.659 | 97.176 | 10.291 |
| 1 | 256 | 64 | 28.609 | 0.617 | 103.746 | 9.639 |
| 8 | 32 | 64 | 16.737 | 0.659 | 776.904 | 10.297 |
| 8 | 256 | 64 | 141.922 | 0.769 | 665.605 | 12.019 |
