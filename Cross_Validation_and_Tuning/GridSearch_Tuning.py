from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# dataset
iris = load_iris()
X, y = iris.data, iris.target

# model
model = RandomForestClassifier(random_state=42)

# parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10],
    'criterion': ['gini', 'entropy']
}

# grid search
grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(X, y)

print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)