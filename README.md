# Retail Customer Analytics - Case Study 1, Part 1

Predicting **marketing campaign response** and **trade channel** on two public
retail datasets, using two models deliberately chosen to differ from the
decision tree, random forest and logistic regression covered in earlier
coursework.

- **M1 - Histogram-based Gradient Boosting** (shallow trees fitted sequentially)
- **M2 - Multi-layer Perceptron** (two hidden layers, 64 and 32 ReLU units)

All results are out-of-fold estimates from stratified 5-fold cross-validation,
with imputation, encoding and scaling fitted **inside** each fold to prevent
leakage.

## Results

| Dataset | Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Brier | Lift@20% |
|---|---|---|---|---|---|---|---|---|
| D1 (n=2,240, 14.9% pos.) | Gradient Boosting | **0.668** | **0.912** | 0.573 | **0.695** | **0.628** | **0.078** | **3.61** |
| D1 | Neural Network (MLP) | 0.624 | 0.879 | **0.592** | 0.617 | 0.604 | 0.086 | 3.44 |
| D2 (n=440, 32.3% pos.) | Gradient Boosting | 0.905 | 0.948 | **0.849** | **0.908** | **0.878** | **0.069** | 2.85 |
| D2 | Neural Network (MLP) | **0.907** | **0.951** | 0.842 | 0.901 | 0.871 | 0.076 | **2.89** |

PR-AUC is the headline metric for D1 because the positive class is only 14.9% of
the base, where accuracy (85.1% for a model that predicts nobody responds) is
meaningless.

## Key findings

1. **Behaviour beats demographics.** The strongest predictors of campaign
   response were prior campaign take-up (0.156) and recency (0.139). Income, age
   and education barely registered.
2. **Targeting works.** Contacting the top-scored 20% of customers reaches about
   3.6x as many responders as contacting 20% at random.
3. **One category identifies account type.** In D2, spend on detergents and
   paper alone accounts for a 0.282 drop in average precision when shuffled, an
   order of magnitude more than any other feature.

The two datasets are complementary rather than contradictory: both show that
*what and how* a customer buys matters more than *who they are*. But category
spend is highly diagnostic of customer **identity** and weakly predictive of
**responsiveness**, so a segmentation feature set should not be reused for
campaign targeting.

## Data

Both datasets are public and are included here unmodified.

| | Source | Size |
|---|---|---|
| D1 - Customer Personality Analysis | [Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) / [direct CSV](https://github.com/nailson/ifood-data-business-analyst-test/blob/master/ml_project1_data.csv) | 2,240 x 29 |
| D2 - Wholesale Customers | [UCI](https://archive.ics.uci.edu/dataset/292/wholesale+customers) / [direct CSV](https://github.com/dphi-official/Datasets/blob/master/Wholesale_customers_data.csv) | 440 x 8 |

## Running it

```bash
pip install -r requirements.txt
python3 analysis.py
```

Expects `data/marketing_campaign.csv` and `data/wholesale_customers.csv`.
Writes the five figures to `figs/` and all metrics to `results.json`.

## Limitations

Both datasets are modest in size, so cross-validated estimates carry meaningful
variance. D2 has only about 28 retail accounts per fold, and the 0.002 PR-AUC
gap between the two models there is well inside noise. Neither dataset was
collected in Australia or covers a recent period, so these results demonstrate
method rather than deployable performance. Neither records protected attributes,
so no fairness audit was possible; that audit would be a prerequisite before any
targeting model was used to allocate discounts.
