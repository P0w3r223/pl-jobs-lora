# Evaluation report (n=142)

Every variant scored by the same pure scorer on the same frozen test set (ADR-0003). API cost priced at claude-haiku-4-5 $1.0/$5.0 per MTok.

| variant | JSON valid | seniority F1 | tech F1 | work-mode F1 | salary(cur/kind/amt) | field F1 | $/1k | p50 s | p95 s |
|---|---|---|---|---|---|---|---|---|---|
| bielik-1.5b-gguf__few | 0.80 | 0.25 | 0.12 | 0.54 | 0.75/0.90/0.75 | 0.23 | - | 21.43 | 103.02 |
| bielik-1.5b-gguf__zero | 0.05 | 0.07 | 0.01 | 0.05 | 0.75/0.87/0.74 | 0.03 | - | 67.90 | 102.58 |
