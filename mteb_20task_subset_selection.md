# MTEB English v2 — Optimal 20-Task Subset Selection

**Goal:** Select 20 tasks (common between embeddinggemma and qwen3) that minimize total
inference time while keeping per-category average scores approximately unchanged.

Both models run the same **40 tasks** across 7 categories.

---

## Selected 20 Tasks

| # | Category | Task | G score | G time (s) | Q score | Q time (s) |
|---|----------|------|--------:|----------:|--------:|----------:|
|  1 | Retrieval          | SCIDOCS                                  | 0.1867 |  11.7 | 0.2415 |  22.6 |
|  2 | Retrieval          | CQADupstackGamingRetrieval               | 0.5997 |  14.9 | 0.6382 |  22.9 |
|  3 | Retrieval          | CQADupstackUnixRetrieval                 | 0.4161 |  24.3 | 0.4970 |  56.3 |
|  4 | Retrieval          | HotpotQAHardNegatives                    | 0.6991 |  36.5 | 0.6677 |  61.4 |
|  5 | Retrieval          | TRECCOVID                                | 0.7585 |  42.6 | 0.9248 | 106.2 |
|  6 | Clustering         | TwentyNewsgroupsClustering.v2            | 0.5230 |   2.1 | 0.5131 |   2.8 |
|  7 | Clustering         | BiorxivClusteringP2P.v2                  | 0.5087 |   3.1 | 0.4768 |   5.0 |
|  8 | Clustering         | MedrxivClusteringS2S.v2                  | 0.4170 |   3.3 | 0.4053 |   4.4 |
|  9 | Clustering         | StackExchangeClustering.v2               | 0.6805 |   5.5 | 0.7114 |   7.7 |
| 10 | Reranking          | AskUbuntuDupQuestions                    | 0.3670 |   3.2 | 0.3749 |   3.7 |
| 11 | STS                | BIOSSES                                  | 0.8344 |   0.4 | 0.8642 |   0.7 |
| 12 | STS                | STS17                                    | 0.8968 |   0.5 | 0.9052 |   0.7 |
| 13 | STS                | STS12                                    | 0.8084 |   0.8 | 0.7484 |   1.1 |
| 14 | Classification     | AmazonCounterfactualClassification       | 0.9125 |   1.4 | 0.9158 |  10.7 |
| 15 | Classification     | MassiveScenarioClassification            | 0.9103 |   2.7 | 0.8312 |  13.4 |
| 16 | Classification     | TweetSentimentExtractionClassification   | 0.6942 |   3.7 | 0.7610 |  26.5 |
| 17 | Classification     | MTOPDomainClassification                 | 0.9916 |  11.9 | 0.9584 |  25.3 |
| 18 | PairClassification | TwitterSemEval2015                       | 0.7744 |   2.1 | 0.7207 |   2.7 |
| 19 | PairClassification | SprintDuplicateQuestions                 | 0.9061 |   3.2 | 0.9743 |   4.0 |
| 20 | Summarization      | SummEvalSummarization.v2                 | 0.3758 |   0.8 | 0.2973 |   1.2 |

---

## Category Distribution

| Category         | Full | Selected |
|------------------|-----:|---------:|
| Retrieval        |   10 |        5 |
| Clustering       |    8 |        4 |
| Reranking        |    1 |        1 |
| STS              |    9 |        3 |
| Classification   |    8 |        4 |
| PairClassification |  3 |        2 |
| Summarization    |    1 |        1 |
| **Total**        | **40** | **20** |

---

## Category Average Preservation

Averages computed over the tasks kept in each subset.
All deviations are within ±3%.

| Category           | G (full) | G (subset) | G shift | Q (full) | Q (subset) | Q shift |
|--------------------|:--------:|:----------:|--------:|:--------:|:----------:|--------:|
| Retrieval          | 0.5189   | 0.5320     |  +2.5%  | 0.6050   | 0.5938     |  −1.9%  |
| Clustering         | 0.5341   | 0.5323     |  −0.3%  | 0.5246   | 0.5267     |  +0.4%  |
| Reranking          | 0.3670   | 0.3670     |   0.0%  | 0.3749   | 0.3749     |   0.0%  |
| STS                | 0.8315   | 0.8465     |  +1.8%  | 0.8163   | 0.8393     |  +2.8%  |
| Classification     | 0.8774   | 0.8771     |   0.0%  | 0.8562   | 0.8666     |  +1.2%  |
| PairClassification | 0.8492   | 0.8403     |  −1.1%  | 0.8532   | 0.8475     |  −0.7%  |
| Summarization      | 0.3758   | 0.3758     |   0.0%  | 0.2973   | 0.2973     |   0.0%  |

---

## Time Savings

| Model          | Full 40 tasks (s) | Selected 20 (s) | Reduction |
|----------------|------------------:|----------------:|----------:|
| embeddinggemma |             841   |            188  |    −77.7% |
| qwen3          |            1340   |            366  |    −72.7% |
| **Combined**   |          **2181** |          **554**|  **−74.6%** |

---

## Dropped Tasks and Rationale

### Retrieval (dropped 5)
| Task | G score | Q score | Combined time (s) | Reason dropped |
|------|--------:|--------:|------------------:|----------------|
| FiQA2018                 | 0.4326 | 0.4723 |  49.4 | Average well covered by kept tasks |
| ClimateFEVERHardNegatives| 0.3021 | 0.3754 |  95.5 | Redundant low-score slot |
| FEVERHardNegatives       | 0.7426 | 0.8656 | 118.9 | High score already represented by TRECCOVID |
| ArguAna                  | 0.5855 | 0.6930 | 314.1 | Mid-score, very expensive |
| Touche2020Retrieval.v3   | 0.4662 | 0.6750 | 604.5 | Most expensive single task in the full set |

### Clustering (dropped 4)
| Task | Combined time (s) | Reason dropped |
|------|------------------:|----------------|
| MedrxivClusteringP2P.v2        | 10.5 | Redundant with MedrxivClusteringS2S kept |
| StackExchangeClusteringP2P.v2  | 26.2 | Redundant with StackExchangeClustering kept |
| ArXivHierarchicalClusteringS2S |  8.0 | Average covered by Biorxiv + TwentyNewsgroups |
| ArXivHierarchicalClusteringP2P | 52.8 | Most expensive clustering task |

### STS (dropped 6)
| Task | Combined time (s) | Reason dropped |
|------|------------------:|----------------|
| STS13         |  1.5 | Redundant with STS12/STS17 |
| STS14         |  2.2 | Redundant with STS12/STS17 |
| STS15         |  2.0 | Redundant with STS12/STS17 |
| STSBenchmark  |  1.5 | Redundant with STS12/STS17 |
| SICK-R        |  2.3 | Redundant with BIOSSES |
| STS22.v2      | 106.2 | 88% of total STS time; low score already anchored by SCIDOCS in retrieval |

### Classification (dropped 4)
| Task | Combined time (s) | Reason dropped |
|------|------------------:|----------------|
| MassiveIntentClassification   |  33.1 | Similar to MassiveScenario kept |
| ToxicConversationsClassification | 39.8 | Covered by TweetSentiment (low accuracy anchor) |
| Banking77Classification       |  54.6 | Covered by Amazon + MTOP (high accuracy anchors) |
| ImdbClassification            |  92.7 | Most expensive classification task |

### PairClassification (dropped 1)
| Task | Combined time (s) | Reason dropped |
|------|------------------:|----------------|
| TwitterURLCorpus | 11.4 | Mid-score; Sprint + TwitterSemEval already span the full range |
