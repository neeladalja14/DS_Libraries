# Data Science Libraries Showcase 🚀

## Overview
This repository provides an introduction to three essential **Python Data Science libraries**:
- **Scikit-Learn** 🤖 - Machine Learning
- **TensorFlow** 🔥 - Deep Learning
- **Matplotlib** 📊 - Data Visualization

## Libraries
### 1️⃣ Scikit-Learn
Scikit-Learn is a powerful library for Machine Learning in Python. It provides simple and efficient tools for:
- Classification (e.g., Logistic Regression, Decision Trees)
- Regression (e.g., Linear Regression, Random Forest)
- Clustering (e.g., K-Means, DBSCAN)
- Model Evaluation (e.g., Cross-Validation, Grid Search)

**Example:**
```python
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

model = LinearRegression()
model.fit(X, y)
print("Predicted:", model.predict([[5]]))  # Output: [10]
```

---
### 2️⃣ TensorFlow
TensorFlow is an open-source deep learning framework by Google. It is widely used for:
- Building Neural Networks
- Image Recognition (CNNs)
- Natural Language Processing (NLP)
- Reinforcement Learning

**Example:**
```python
import tensorflow as tf

x = tf.constant(5)
y = tf.constant(3)
result = tf.add(x, y)
print("TensorFlow Addition:", result.numpy())  # Output: 8
```

---
### 3️⃣ Matplotlib
Matplotlib is a visualization library used to create static, animated, and interactive plots.

**Example:**
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, label='Sine Wave')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Line Plot')
plt.legend()
plt.show()
```

## Installation
Install all libraries using:
```bash
pip install scikit-learn tensorflow matplotlib
```

## Contributions
Feel free to contribute by adding more examples, improving documentation, or suggesting new features. Open a PR! 🎯

## License
This project is licensed under the **MIT License**. 📝
