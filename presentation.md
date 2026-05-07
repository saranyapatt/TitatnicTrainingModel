# Titanic Survival Prediction
## KNN & Random Forest — Kaggle Competition
*~2-3 minute presentation*

---

## Slide 1: Title

**Titanic Survival Prediction**
K-Nearest Neighbors vs Random Forest

- Platform: Kaggle Competition
- Dataset: 1,309 passengers (891 train / 418 test)
- Task: Binary classification — Did the passenger survive?

---

## Slide 2: The Dataset

**891 training passengers, 12 original features**

| Feature | What It Tells Us |
|---|---|
| Pclass | Ticket class (1st / 2nd / 3rd) — proxy for wealth |
| Sex | Strong survival predictor ("women and children first") |
| Age | 19% missing — required imputation |
| SibSp / Parch | Family size on board |
| Fare | Ticket price — correlated with class |
| Embarked | Port of boarding (C / Q / S) |
| Cabin | 77% missing — dropped |

**Target:** Survived (0 = No, 1 = Yes)
Overall survival rate: **38%** (342 out of 891)

---

## Slide 3: Missing Data Challenge

**Three columns had missing values:**

| Column | Missing | Strategy |
|---|---|---|
| Age | 19% (177 rows) | Imputed by title-group median |
| Embarked | 0.2% (2 rows) | Filled with mode (S) |
| Fare | 1 row (test only) | Filled with title-group median |
| Cabin | 77% | Dropped entirely |

> Key insight: Instead of using a single global Age median,
> we grouped passengers by title (Mr, Mrs, Master, etc.)
> and used each group's median — more accurate.

---

## Slide 4: Feature Engineering — The Title Trick

**Extracted title from passenger names using regex:**

```
"Braund, Mr. Owen Harris"  →  Mr
"Heikkinen, Miss. Laina"   →  Miss_Mrs
"Becker, Master. Richard"  →  Master
```

**Grouped 18 rare titles into 5 categories:**

| Group | Titles |
|---|---|
| Mr | Mr |
| Miss_Mrs | Miss, Mrs, Mlle, Ms, Mme |
| Master | Master |
| Royalty | Lady, Sir, Jonkheer, Don, Countess |
| Officer | Dr, Rev, Col, Major, Capt |

One-hot encoded → used to impute Age by group

---

## Slide 5: Full Preprocessing Pipeline

```
Raw CSV
  ↓
Extract title from Name  →  group  →  one-hot encode
  ↓
Impute Age by title-group median
  ↓
Fill Embarked (mode)  |  Fill Fare (title median, test only)
  ↓
Encode Sex: male=0, female=1
  ↓
One-hot encode Embarked (C / Q / S)
  ↓
Drop: Cabin, Ticket, PassengerId, Name
  ↓
[KNN only] StandardScaler normalization
  ↓
Final features: Pclass, Sex, Age, SibSp, Parch, Fare,
                Embarked_*, Title_*   (13 features total)
```

---

## Slide 6: Model 1 — K-Nearest Neighbors

```python
KNeighborsClassifier(n_neighbors=5)
```

**How it works:**
- For each test passenger, find the 5 most similar training passengers (by Euclidean distance)
- Predict survival by majority vote among those 5 neighbors

**Key parameter: k = 5**
- Small k → sensitive to noise
- Large k → smoother but may miss local patterns
- k=5 is a common starting point; tuning with cross-validation is the next step

**Important:** Features were scaled with `StandardScaler` before fitting.
KNN is distance-based — without scaling, `Fare` (0–512) would dominate `Pclass` (1–3).

**Kaggle Score: 77%**

---

## Slide 7: Model 2 — Random Forest

```python
RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
```

**How it works:**
- Builds 100 decision trees, each trained on a random bootstrap sample of the data
- Each split uses a random subset of features (reduces correlation between trees)
- Final prediction = majority vote across all 100 trees

**Key parameters:**
| Parameter | Value | Reason |
|---|---|---|
| n_estimators | 100 | More trees = more stable predictions |
| max_depth | 5 | Limits tree depth to prevent overfitting |
| random_state | 42 | Ensures reproducible results |

**No feature scaling needed** — tree-based models split on thresholds, not distances.

**Kaggle Score: 76%**

---

## Slide 8: Results & Takeaways

**Final Scores**

| Model | Kaggle Public Score |
|---|---|
| KNN (k=5) | **77%** |
| Random Forest (100 trees, depth 5) | **76%** |

**What worked well:**
- Title-based Age imputation (domain-aware, not lazy)
- Correct use of StandardScaler for KNN, skipped for RF
- One-hot encoding avoided false ordinal assumptions

**What could improve the score further:**
- Cross-validation to tune hyperparameters (k, n_estimators, max_depth)
- Feature engineering: FamilySize = SibSp + Parch + 1, IsAlone flag
- Ensemble: combine KNN + RF predictions

---
*End of presentation*
