"""Stratified five-fold evaluation of the project's RBF SVC."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA = Path(__file__).with_name('dataset.csv')
SEED = 42
FOLDS = 5


def main():
    data = pd.read_csv(DATA).head(5000)
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    scores = {'accuracy': [], 'macro-F1': []}
    for train, test in cv.split(X, y):
        model = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
        model.fit(X.iloc[train], y.iloc[train])
        pred = model.predict(X.iloc[test])
        scores['accuracy'].append(accuracy_score(y.iloc[test], pred))
        scores['macro-F1'].append(f1_score(y.iloc[test], pred, average='macro'))
    print(f'Rows: {len(data)}; folds: {FOLDS}; seed: {SEED}')
    for name, values in scores.items():
        print(f'{name}: {np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}')


if __name__ == '__main__':
    main()
