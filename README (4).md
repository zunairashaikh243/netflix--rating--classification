# 🎬 Netflix Audience Rating Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Task](https://img.shields.io/badge/Task-Multi--Class%20Classification-E50914)
![Best model](https://img.shields.io/badge/Best%20Model-Tuned%20Random%20Forest-141414)

Predict the **audience category** of a Netflix title (**Kids, Family, Teens, Adults**) from its content attributes using **Decision Trees** and **Random Forests**, then tune, compare and evaluate the models.

> Dataset: 8,790 Netflix titles (movies and TV shows) with type, director, country, release year, duration and genres.

---

## 📌 Results at a glance

| Model (held-out test set, 1,742 titles) | Accuracy | Macro F1 |
|---|:---:|:---:|
| Majority-class baseline | 0.460 | 0.158 |
| Decision Tree (default) | 0.548 | 0.526 |
| Decision Tree (**tuned**) | 0.596 | 0.573 |
| Logistic Regression | 0.622 | 0.586 |
| Random Forest (default) | 0.617 | 0.587 |
| Gradient Boosting | **0.641** | 0.602 |
| Random Forest (**tuned**) 🏆 | 0.627 | **0.612** |

The tuned Random Forest is selected by **macro-F1** because the classes are imbalanced (Adults ≈ 46%, Kids ≈ 10%). Gradient Boosting edges it on raw accuracy, but is weaker on the smaller classes.

![Model comparison](images/06_model_comparison.png)

---

## 🌐 Live demo

An interactive **Streamlit** app lets you describe a title (type, runtime, genres, countries, year) and see the predicted audience category with a probability for each class.

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Deploy it for free (get a public link):**
1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub and click **New app**.
3. Select this repo, branch `main`, main file `app.py`, then **Deploy**.
4. Paste the resulting link here: `Live demo: <your-app-link>`

> If you ever see a scikit-learn version error when loading the model, run `python main.py` once to regenerate `models/best_model.joblib` with your installed version.

---

## 🧭 Task workflow

| Step | What was done | Code |
|---|---|---|
| **1. Analyze rating categories** | 14 raw ratings grouped into 4 audience categories, unrated titles dropped, class balance and genre/type patterns explored | `src/eda.py` |
| **2. Prepare training datasets** | 53 engineered features, stratified 80/20 split | `src/data_preprocessing.py` |
| **3. Train classification models** | Baseline, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting with 5-fold stratified CV | `src/train.py` |
| **4. Optimize performance** | `GridSearchCV` for the tree, `RandomizedSearchCV` for the forest, scored on macro-F1 | `src/train.py` |
| **5. Evaluate accuracy** | Accuracy, precision, recall, F1, confusion matrices, feature importance, learning curve | `src/evaluate.py` |

### Step 1: Rating categories

| Category | Raw ratings |
|---|---|
| Kids | TV-Y, TV-Y7, TV-Y7-FV, TV-G, G |
| Family | TV-PG, PG |
| Teens | TV-14, PG-13 |
| Adults | TV-MA, R, NC-17 |
| *(dropped)* | NR, UR (not rated, 82 titles) |

![Raw ratings](images/01_raw_ratings.png)
![Class balance](images/02_class_balance.png)
![Genre heatmap](images/04_genre_heatmap.png)

### Step 2: Features (53 in total, none use the rating itself)

- **Content:** movie vs TV show, runtime (minutes), number of seasons
- **Time:** release year, year and month added, years between release and arrival on Netflix
- **People and place:** has director, number of directors, number of countries, 12 top-country flags
- **Genres:** number of genres, 30 top-genre flags

### Step 3 and 4: Training and tuning

5-fold cross-validated scores on the training set:

| Model | CV Accuracy | CV Macro F1 |
|---|:---:|:---:|
| Majority baseline | 0.460 | 0.158 |
| Decision Tree | 0.547 | 0.539 |
| Logistic Regression | 0.629 | 0.592 |
| Random Forest | 0.628 | 0.602 |
| Gradient Boosting | 0.646 | 0.610 |

Best hyperparameters found (full list in `reports/best_params.json`):

| Model | Best settings |
|---|---|
| Decision Tree | `criterion=gini`, `max_depth=None`, `min_samples_leaf=10`, `min_samples_split=30` |
| Random Forest | `n_estimators=350`, `max_depth=12`, `min_samples_leaf=5`, `max_features=0.3`, `class_weight=balanced_subsample` |

![Tuning effect](images/09_tuning_effect.png)
![Depth curve](images/11_tree_depth_curve.png)

### Step 5: Evaluation

Per-class results for the tuned Random Forest:

| Class | Precision | Recall | F1 | Support |
|---|:---:|:---:|:---:|:---:|
| Kids | 0.821 | 0.735 | 0.776 | 181 |
| Family | 0.415 | 0.426 | 0.421 | 230 |
| Teens | 0.542 | 0.537 | 0.539 | 529 |
| Adults | 0.705 | 0.721 | 0.713 | 802 |

![Confusion matrix](images/12_confusion_matrix_best.png)
![Feature importance](images/08_feature_importance.png)

---

## 💡 Key insights

1. **Kids content is easiest to spot** (F1 0.78), mostly thanks to the *Kids' TV* and *Children & Family Movies* genre tags.
2. **Errors are between neighbouring categories**: Teens ↔ Adults and Family ↔ Teens are confused far more than distant pairs. Almost no Teens or Adults titles are predicted as Kids.
3. **Hyperparameter tuning helped**: macro-F1 rose from 0.526 to 0.573 for the Decision Tree and from 0.587 to 0.612 for the Random Forest.
4. **Ensembles beat a single tree**: an un-pruned tree over-fits, and averaging many trees fixes most of that.
5. **There is a ceiling at about 63% accuracy**: ratings depend on violence, language and themes, which are not in this dataset. Adding text features (descriptions, cast) is the natural next step.

---

## 🚀 How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (EDA, training, tuning, evaluation)
#    Note: the tuning step takes a few minutes on a single CPU core
python main.py

# 3. Try the saved model on new titles
python predict.py
```

Outputs are written to `images/`, `reports/` and `models/`. To explore step by step, open `notebooks/Netflix_Rating_Classification.ipynb`.

## 🗂️ Project structure

```
netflix-rating-classification/
├── data/Dataset.csv                  # Netflix titles
├── src/
│   ├── config.py                     # paths, seed, rating groups, theme
│   ├── data_preprocessing.py         # target grouping, feature engineering, split
│   ├── eda.py                        # Step 1 charts
│   ├── inference.py                  # load model and predict new titles
│   ├── train.py                      # models, cross-validation, tuning
│   ├── evaluate.py                   # metrics and evaluation charts
│   └── plot_style.py                 # shared plot theme
├── notebooks/Netflix_Rating_Classification.ipynb
├── images/                           # all generated charts
├── reports/                          # CSV/JSON/TXT metrics
├── models/best_model.joblib          # saved best model
├── app.py                            # Streamlit live demo
├── main.py                           # run everything
├── predict.py                        # predict new titles
├── .streamlit/config.toml            # demo theme
└── requirements.txt
```

## 🛠️ Tech stack
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, joblib, Streamlit

## 🔭 Future improvements
- Add TF-IDF features from titles and descriptions
- Try XGBoost / LightGBM and probability calibration
- Use SHAP for per-prediction explanations
- Treat the four categories as ordered (ordinal classification)
