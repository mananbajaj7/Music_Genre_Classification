# Music_Genre_Classification

# 🎶 Music Genre Classification with Audio Features

This project explores multi-class classification of music genres using 50,000 randomly selected songs and their audio features obtained from Spotify. The objective was to predict the genre of each song using classification models and various preprocessing techniques.

---

## 📁 Dataset Overview

* **Source**: Spotify API (via `musicData.csv`)
* **Observations**: 50,000 songs
* **Features**: Tempo, energy, valence, loudness, danceability, duration, and more
* **Target**: Music genre (10 classes)

---

## 🧹 Data Cleaning & Preprocessing

* Dropped 5 rows with missing data
* Replaced placeholder "?" in `tempo` with the mode
* Set negative `duration_ms` to NaN and imputed with mode
* Dropped irrelevant columns: `instance_id`, `artist_name`, `track_name`, `obtained_date`
* Standardized all numeric features using Z-score normalization
* One-hot encoded categorical columns (`mode`, `key`)
* Transformed `music_genre` into numerical categorical labels
* Created a stratified train/test split with 500 samples per genre in the test set (10 genres x 500 = 5000 test samples)

---

## 📉 Dimensionality Reduction

* Compared **PCA** and **UMAP** for feature reduction
* PCA was faster and retained with 3 components (using Kaiser criterion)
* UMAP and PCA failed to visibly separate genres well in 2D projections

---

## 📊 Clustering Analysis (Exploratory)

* Used K-means clustering on PCA-reduced data
* Silhouette scores indicated 2 clusters — poor fit for 10-genre classification
* Concluded K-means not suitable for this dataset

---

## 🤖 Model Evaluation

Tested multiple classification models:

* ✅ **XGBoost** (Best)
* Random Forest
* AdaBoost
* Neural Network

| Model         | Accuracy | AUC    |
| ------------- | -------- | ------ |
| XGBoost       | 0.5766   | 0.9307 |
| Neural Net    | 0.5634   | 0.9193 |
| AdaBoost      | \~0.53   | \~0.89 |
| Random Forest | \~0.55   | \~0.91 |

✅ XGBoost was the top performer on both PCA and full feature sets.

---

## 🔧 Hyperparameter Tuning

* Used `RandomizedSearchCV` with cross-validation
* Optimized parameters for XGBoost
* Final test AUC: **0.933**

---

## 💡 Extra Credit: Genre Similarity Visualization

* Plotted **cosine similarity** between genres
* Visualized genre overlap contributing to classification difficulty
* Highlighted real-world complexity of genre prediction tasks

---

## 📌 Conclusion

* XGBoost's structure makes it ideal for complex, non-linear datasets like audio features
* Genre overlap explains limited classification accuracy
* PCA + XGBoost with hyperparameter tuning was the most effective combination

---

## 📎 Tools & Techniques Used

* Python (Pandas, Scikit-learn, XGBoost)
* Matplotlib, Seaborn
* PCA, UMAP
* RandomizedSearchCV
* One-Hot Encoding, Z-score Normalization
* Stratified Train-Test Split
* K-means Clustering (exploratory)
* Multi-class Classification
* AUC, ROC Curve, Silhouette Score
* Hyperparameter Tuning with Cross-validation

---

## 🧠 Project Type

Capstone project for **Introduction to Machine Learning**, NYU

> *"This was a real-world classification challenge that incorporated full pipeline modeling under time constraints — great portfolio material."*
