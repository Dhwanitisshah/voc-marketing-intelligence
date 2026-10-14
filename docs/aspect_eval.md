# Aspect-level sentiment: before / after evaluation

**What was compared.** How a keyword aspect mention gets its sentiment:

| Mode | Behaviour |
|---|---|
| `review` | every mention inherits the whole review's sentiment (the original behaviour) |
| `sentence` | only the sentence containing the keyword is scored with the same RoBERTa model (class probabilities averaged if several sentences mention the aspect); negated positive keywords ("not easy", "no value", "not worth") count as negative mentions |
| `sentence+clause` | as `sentence`, but also cut at "but" / "however" / "although" / ";" and score only the keyword's clause (`ASPECT_SPLIT_CLAUSES = True`, an ablation; **off by default**) |

**Data.** `reviews_171e719d99` (3,000 Datafiniti Amazon reviews). Aspects audited: `ux` and `price`.

**Labels: these are draft labels, not ground truth.** `data/eval/aspect_audit.csv` has 136 rows. **0 have a `human_label`; all 136 `draft_label` values were written by Claude**, so every precision figure below is "agreement with Claude's labels" until you fill `human_label` (the script prefers `human_label` over `draft_label` automatically). The 136 rows are every mention that was negative in at least one mode (`found_in` says which): the original 75 review-level negatives (40 ux + 35 price), plus the 61 extra mentions that only the sentence variants flagged. Labelling those extras too is what keeps the precision of the new modes honest; without them, mentions the new modes added would have been scored as if they didn't exist.

Label meaning: `genuine` = the reviewer criticises the aspect itself; `inherited` = the aspect is mentioned but not criticised (praise or neutral), the negativity comes from elsewhere in the review; `false_trigger` = the keyword matched something that isn't the aspect (idiom, a competitor's price, a build-quality "cheap", …). Precision = `genuine` / negative mentions.

One draft label was revised while labelling the second batch, for consistency: price #1938 ("Didn't work more than 4 months. Don't waste your money.") went from `inherited` to `genuine`, because an explicit "waste of money" is a value judgement whatever triggered it.

## Results

| Aspect | Mode | Negative mentions | Labelled | Genuine | Precision | Impact rank | Close to |
|---|---|---:|---:|---:|---:|---:|---|
| ux | review | 40 | 40 | 30 | 75% | 1 | price |
| ux | sentence | 52 | 52 | 43 | 83% | 1 | price |
| ux | sentence+clause | 50 | 50 | 42 | 84% | 1 | price |
| price | review | 35 | 35 | 12 | 34% | 2 | ux |
| price | sentence | 49 | 49 | 20 | 41% | 2 | ux |
| price | sentence+clause | 44 | 44 | 25 | 57% | 2 | ux |

"Impact rank" is the aspect's rank among all six aspects by negative-mention count. "Close to" is the new statistical-closeness flag (counts whose ±√n intervals overlap).

### The 7 known false UX cases

The seven false UX mentions found in the first audit sample (seed 42).

| Review | Draft label | review | sentence | sentence+clause |
|---|---|---|---|---|
| #170 | inherited | still negative | still negative | still negative |
| #229 | inherited | still negative | still negative | fixed |
| #413 | inherited | still negative | fixed | fixed |
| #593 | inherited | still negative | fixed | fixed |
| #1345 | false_trigger | still negative | fixed | fixed |
| #1399 | false_trigger | still negative | fixed | fixed |
| #1585 | false_trigger | still negative | still negative | still negative |
| **Fixed** | | **0/7** | **4/7** | **5/7** |

Why three survive sentence mode:

- **#229** "The sound is not good **but** it was easy to set up." One sentence, so the bad sound drags the "easy" mention negative. Clause splitting fixes it.
- **#170** Two sentences mention apps: "Apps are ok…" and "…it won't download apps to the sd card!". The second is genuinely negative about apps; the draft label (`inherited`) arguably undersells it.
- **#1585** "Not enough memory to hold kids apps." The sentence really is negative; the complaint is storage and "apps" is incidental. Sentence-level scoring can't tell which noun the negativity is about (that needs a target-aware ABSA model).

### Runtime added (3,000 reviews, model already loaded, 24-core CPU)

| Mode | Aspect-sentiment step |
|---|---:|
| review | 0.0 s |
| sentence | 89.7 s |
| sentence+clause | 91.1 s |

About 1,500 sentences go through RoBERTa (whole-review sentences reuse the review-level score instead of being re-scored; sentences are batched by length). The cost is model inference, which was measured to be insensitive to batch size, so it scales roughly linearly with the number of aspect mentions.

## What this does and does not show

1. **Sentence scoring helps, but modestly, and mostly by finding more rather than by removing noise.**
   - UX precision rises 75% → 83% and genuine UX complaints found rise 30 → 43. The extra negatives (40 → 52) are mostly real complaints sitting inside otherwise positive reviews ("Love it. … my only complaint is that the interface is slow sometimes") that review-level scoring could never see.
   - Within the labelled pool (79 genuine mentions across both aspects), review mode finds 42 (53%), sentence 63 (80%), sentence+clause 67 (85%). That pool only contains mentions that were negative in some mode, so treat the recall figures as indicative, not absolute.
2. **Price is still mostly noise after sentence scoring** (41%; 57% with clauses). The remaining false negatives are largely praise that the sentiment model reads as negative because of phrasing: "For the price I ended up getting it for, there's no arguement", "can't beat the prices", "I can't believe it's so cheap now". Negation handling only helps *positive-polarity* keywords; "price" is neutral, so it is left to the model, which is trained on tweets and misreads these. Price also carries idiom false triggers ("not that I have money to waste", cable's price, an iPad's price).
3. **The impact ranking did not change, and shouldn't be read as settled.** ux is #1 and price #2 in every mode, and the closeness flag correctly says they are statistically close on raw counts (52 vs 49). But counting only labelled-genuine mentions, ux has 43 and price 20, so on this sample ux is a clearly bigger problem than the raw counts suggest. The raw counts understate the gap because price's negatives are mostly not price complaints.
4. **Clause splitting looks better on every measure here** (price precision 41% → 57%, price genuine 20 → 25, one more known case fixed) at a cost of one UX genuine mention (43 → 42). It is off by default because the requested behaviour was sentence-level; flip `ASPECT_SPLIT_CLAUSES` in `config.py` to use it. Datasets analysed under a different setting are rebuilt automatically on load.

## Caveats

- All labels are drafts by the same author as the code under test; a second human pass (`human_label`) is the thing most likely to move these numbers.
- Small samples: 40–52 mentions per UX cell, 35–49 per price cell. A ±10-point swing in precision is within noise.
- One dataset (Amazon electronics). The lexicon was built for delivery/service language, which is part of why price and UX behave this way here.

## Reproduce

```bash
python scripts/audit_aspect.py --export data/eval/aspect_audit.csv   # original 75 review-level negatives
python scripts/eval_aspects.py --append-unlabelled                   # add mentions only the new modes flag
python scripts/eval_aspects.py                                       # prints the tables above
```

`--out FILE` saves the tables to a file; don't point it at this document, which also holds the narrative.
