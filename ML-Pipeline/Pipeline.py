# Machine Learning me pipeline =
# step-by-step automated flow, such as:

#  Data Cleaning
#  Feature Scaling
#  Encoding
#  Dimensionality Reduction
#  Model Training
#  Evaluation
#  Deployment

# > Ek code se saare steps automatic ho jaate hain
# No manual calling of preprocess → train → test → again repeat

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),     # Scaling
    ('pca', PCA(n_components=2)),     # Dimensionality Reduction
    ('model', SVC(kernel='rbf'))      # Training Model
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))