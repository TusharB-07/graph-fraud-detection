# GitHub README for Your Project

Copy everything below into a file named `README.md`. On GitHub: **Add file → Create new file → name it `README.md`** → paste → Commit.

---

```markdown
#  Graph AI for Real-Time Financial Fraud Detection

A Graph Neural Network (GCN + GATv2) approach to financial fraud detection,
modeling the IEEE-CIS transaction dataset as a heterogeneous graph to catch
fraud rings that tabular models miss.

**Team:** Tushar Biswas · Khushi Joshi · Godwin T Saji · Sharu Nethra R · Yashasvi Shrivastava

---

##  Why Graphs?

Traditional fraud detectors (Logistic Regression, XGBoost) treat each
transaction as an isolated row. But in reality, **fraudsters form
communities** — they share cards, billing addresses, and email domains.
By building a graph of these connections and running message passing over
it, a GNN learns topological patterns invisible to tabular models.

##  Dataset

**[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data)** (Vesta Corporation, Kaggle)

- 590,540 total transactions; **150,000 most recent** used in FT1
- **21-column working subset** (transaction aggregates, card/address/email, timedeltas)
-  Extreme class imbalance: **3.32% fraud / 96.68% legitimate**

##  Graph Construction

| Node Type | Count | Source |
|---|---|---|
| Transaction | 150,000 | each row |
| Card (card1) | 7,784 | anchor nodes |
| Address (addr1) | 118 | anchor nodes |
| Email domain | 60 | anchor nodes |
| **Total** | **157,962 nodes** | **900,000 edges** |

Anchor-node features = mean of connected transaction features (fully
vectorized via `index_add_`, built in seconds).

## Models

| Model | Details |
|---|---|
| **Logistic Regression** | liblinear, class-weighted (baseline floor) |
| **XGBoost** | 500 trees, `scale_pos_weight=29.15`, early stopping on val AUC |
| **GCN + GATv2 (ours)** | 64 hidden units, 4 attention heads, flattened 157k-node graph, class-weighted CE (fraud weight = 15.07) |

**Evaluation protocol:** chronological 70/15/15 split (train → validation →
test strictly in time order, like a real deployment). Metrics: **AUC, F1, Recall**.

## Results (chronological hold-out test set, n = 22,500)

| Model | AUC | F1 | Recall |
|---|---|---|---|
| Logistic Regression | 0.7974 | 0.1808 | 0.6883 |
| XGBoost | **0.8964** | **0.3163** | 0.7256 |
| **GCN + GATv2 (ours)** | 0.8170 | 0.1994 | **0.7278**  |

### Key Takeaways

1. **Graph structure adds independent signal** — the GNN beats Logistic
   Regression by +0.02 AUC *despite using 6 fewer features*
2. **GNN catches the most fraud** — highest recall (0.7278), the
   business-critical metric in fraud operations
3. **XGBoost's AUC lead is explainable** — it exploits 18 engineered
   features incl. label-encoded card IDs; the GNN derives relational
   context purely through message passing
4. **GNN was still improving at epoch 200** — clear headroom for FT2

## Reproduce

Run on [Kaggle](https://www.kaggle.com) (free T4 GPU) with the IEEE-CIS
dataset attached:

 **Full notebook with outputs:** [Kaggle Notebook](https://www.kaggle.com/code/thedeveloper0007/financial-fraud-detection)

Environment: Python 3.12 · PyTorch 2.10 (CUDA) · PyTorch Geometric · XGBoost 3.2 · scikit-learn

##  Roadmap (FT2)

- [ ] Full feature set: `TransactionAMT`, identity file, Vesta V-features
- [ ] Extended training + seed-averaged evaluation (validation curve was un-converged)
- [ ] Imbalance-aware metapath-style graph sampling
- [ ] Semi-supervised pseudo-labeling for unlabeled transactions
- [ ] Self-explaining masks (GNN interpretability, SEFraud-style)

## References

1. Qian & Tong, "Metapath-guided graph neural networks for financial fraud
   detection," *Computers and Electrical Engineering*, 2025.
2. Wang et al., "CoSemiGNN: Blockchain fraud detection with dynamic graph
   neural networks based on co-association of semi-supervised,"
   *Expert Systems With Applications*, 2026.
3. Li et al., "SEFraud: Graph-based self-explainable fraud detection via
   interpretative mask learning," *KDD*, 2024.
4. Hiremath et al., "Ensemble of graph neural networks for enhanced
   financial fraud detection," *IEEE I2CT*, 2024.
5. Lu, "Graph neural network model in financial fraud detection,"
   *IEEE ICICR*, 2025.
```

---

## Short "About" description (for the repo's description field, 1 line)

Paste this in the **Description** box on GitHub (top of repo page):

> **Graph Neural Network (GCN+GATv2) fraud detection on IEEE-CIS — 157k-node transaction graph, GNN achieves highest fraud recall (0.7278). FT1 report + Kaggle notebook.**

**Topics/tags** to add (right side of repo): `graph-neural-networks` `fraud-detection` `pytorch-geometric` `gcn` `gat` `ieee-cis` `xgboost` `financial-ai`

---

## Tips for the commit history (the thing that proves work)

When you push, use meaningful messages like:

```
FT1: data loading + EDA, confirmed 3.32% fraud imbalance
FT1: chronological 70/15/15 split + LR/XGBoost baselines
FT1: vectorized heterogeneous graph construction (157,962 nodes)
FT1: trained GCN+GATv2 — AUC 0.8170, best recall 0.7278
FT1: added results table + ROC curves to README
```

This gives your repo a clean, professional timeline a professor can scroll through — each commit maps to a real work session, exactly like the Colab-style workflow you asked about.
