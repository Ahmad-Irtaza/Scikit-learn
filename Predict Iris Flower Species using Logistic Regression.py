# Step 1: Required libraries import karo
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Step 2: Dataset load karo
# Iris dataset mein 3 types ke flowers hain: setosa, versicolor, virginica
iris = load_iris()
X = iris.data            # Features: sepal/petal length/width
y = iris.target          # Target classes: 0, 1, 2

# Step 3: Data ko training aur testing mein split karo
# 80% data model training ke liye, 20% test ke liye
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Logistic Regression model banao
model = LogisticRegression(max_iter=200)  # Max iteration zyada rakhein taake training complete ho

# Step 5: Model ko training data se fit (train) karo
model.fit(X_train, y_train)

# Step 6: Model ko test data pe lagao aur prediction nikaalo
y_pred = model.predict(X_test)

# Step 7: Model ki accuracy aur performance evaluate karo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)

# Final results print karo
print("✅ Accuracy:", accuracy)
print("\n📋 Classification Report:\n", report)
