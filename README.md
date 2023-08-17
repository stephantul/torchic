<p align="center">
  <picture>
    <source srcset="pictures/torchic_logo.png">
    <img alt="Hugging Face Transformers Library" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg" width="160" height="160" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

# torchic

Simple linear model in Pytorch, with a scikit-learn compatible API.

It has the following features:
* Scikit-learn like API (i.e., using `fit` and `predict`)
* Supports numpy arrays and torch tensors out of the box
* Automatically converts your tensors between devices
* Almost nothing is configurable

## Example

The example below classifies 20 newsgroups, which is pre-vectorized using a CountVectorizer, courtesy of scikit-learn. This example requires that scikit-learn is installed.

```python
import numpy as np

from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from torchic import Torchic

# NOTE: change this to 'cuda' or 'mps' if you want acceleration.
DEVICE = "cpu"

X, y = fetch_20newsgroups_vectorized(return_X_y=True, remove=("headers", "footers"), subset="train")
X = X[y < 10]
y = y[y < 10]
X = np.asarray(X.todense())

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=44, test_size=.1)
n_features, n_labels = X_train.shape[1], len(set(y))

# Torchic stuff begins here.
t = Torchic(n_features, n_labels, learning_rate=1e-4).to(DEVICE)
t.fit(X_train, y_train, batch_size=128)

pred = t.predict(X_test)

print(precision_recall_fscore_support(y_test, pred, average="macro"))
```

## TODO:

* Add docstrings
* Add additional unit tests
