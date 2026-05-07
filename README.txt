# Titanic Survival Prediction — Kaggle Competition

Predicting passenger survival on the Titanic using K-Nearest Neighbors and Random Forest classifiers.

**Kaggle Scores**
| Model | Public Score |
|---|---|
| KNN (k=5) | 77% |
| Random Forest | 76% |

---

## Dataset

Source: [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)

| Split | Rows |
|---|---|
| Training set | 891 |
| Test set | 418 |

**Features**

| Column | Type | Description |
|---|---|---|
| Pclass | int | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| Sex | str | Passenger sex |
| Age | float | Age in years (19% missing in train) |
| SibSp | int | # siblings / spouses aboard |
| Parch | int | # parents / children aboard |
| Fare | float | Passenger fare |
| Embarked | str | Port of embarkation: C, Q, or S (0.2% missing) |
| Cabin | str | Cabin number (77% missing — dropped) |
| Ticket | str | Ticket number (dropped) |
| Name | str | Passenger name (used for title extraction, then dropped) |

**Target:** `Survived` (0 = No, 1 = Yes) — overall survival rate: ~38%

---

## Data Preprocessing

1. **Title extraction** — Parse title (Mr, Mrs, Miss, Master, etc.) from `Name` using regex
   Group into 5 categories: `Mr`, `Miss_Mrs`, `Master`, `Royalty`, `Officer`
   One-hot encode as `Title_*` columns

2. **Age imputation** — Fill missing `Age` with the median age for each title group
   (e.g., Master median ≠ Mr median, making this more accurate than a global median)

3. **Embarked imputation** — Fill 2 missing values with mode (`S`)

4. **Fare imputation** — Fill 1 missing test value with the title-group median

5. **Encoding**
   - `Sex`: binary (`male` → 0, `female` → 1)
   - `Embarked`: one-hot encoded (`Embarked_C`, `Embarked_Q`, `Embarked_S`)

6. **Dropped columns** — `Cabin` (77% missing), `Ticket`, `PassengerId`, `Name`

7. **Feature scaling** — `StandardScaler` applied for KNN only (not needed for Random Forest)

---

## Models

### K-Nearest Neighbors (`TitanicKNNPrediction.py`)
```python
KNeighborsClassifier(n_neighbors=5)
```
- Features scaled with `StandardScaler` before fitting (required for distance-based models)
- Trained on full training set; predictions generated for test set

### Random Forest (`TitanicRandomForsestPrediction.py`)
```python
RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
```
- 100 decision trees, max depth 5 (prevents overfitting)
- No feature scaling needed (tree-based model is scale-invariant)
- `random_state=42` ensures reproducible results

---

## How to Run

1. Clone this repo and install dependencies:
   ```bash
   pip install pandas scikit-learn python-dotenv
   ```

2. Create a `.env` file with your dataset paths:
   ```
   PATH2TRAIN=/path/to/train.csv
   PATH2TEST=/path/to/test.csv
   ```

3. Run either model:
   ```bash
   python TitanicKNNPrediction.py
   python TitanicRandomForsestPrediction.py
   ```

4. Output: `submission.csv` ready to upload to Kaggle

---

## Results

| Model | Kaggle Public Score |
|---|---|
| KNN (k=5) | **77%** |
| Random Forest (100 trees, depth 5) | **76%** |

Both models use the same preprocessing pipeline. The marginal KNN advantage may reflect
the benefit of feature scaling combined with the relatively small dataset size (891 rows).
