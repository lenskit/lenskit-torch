---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Explicit Feedback Demo

This notebook demonstrates and measures explicit-feedback performance for the Torch-based algorithms.

+++

## Setup

Load modules:

```{code-cell} ipython3
import sys
import logging
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
```

```{code-cell} ipython3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

```{code-cell} ipython3
from lenskit import batch
from lenskit.metrics.predict import rmse
from lenskit.algorithms.bias import Bias
from lenskit.algorithms.als import BiasedMF
from lktorch.biasedmf import TorchBiasedMF
```

```{code-cell} ipython3
from tqdm.notebook import tqdm
tqdm.pandas()
```

## MovieLens 100K

```{code-cell} ipython3
train_100k = pd.read_parquet('data/ml-100k/train.parquet')
test_100k = pd.read_parquet('data/ml-100k/test.parquet')
```

### Bias Model

```{code-cell} ipython3
bias = Bias()
bias.fit(train_100k)
```

```{code-cell} ipython3
bias_preds = batch.predict(bias, test_100k)
bias_preds
```

```{code-cell} ipython3
rmse(bias_preds['prediction'], bias_preds['rating'])
```

### ALSModel

```{code-cell} ipython3
als = BiasedMF(25)
als.fit(train_100k)
```

```{code-cell} ipython3
als_preds = batch.predict(als, test_100k)
als_preds
```

```{code-cell} ipython3
rmse(als_preds['prediction'], als_preds['rating'])
```

### Biased MF

```{code-cell} ipython3
tmf = TorchBiasedMF(25, epochs=20)
tmf.fit(train_100k)
```

```{code-cell} ipython3
tmf_preds = batch.predict(tmf, test_100k)
tmf_preds
```

```{code-cell} ipython3
rmse(tmf_preds['prediction'], tmf_preds['rating'])
```

### Integrate Predictions


```{code-cell} ipython3
preds_100k = pd.concat({
    'Bias': bias_preds,
    'ALS': als_preds,
    'Torch BiasedMF': tmf_preds,
}, names=['algo'])
preds_100k = preds_100k.reset_index('algo').reset_index(drop=True)
preds_100k.head()
```

```{code-cell} ipython3
preds_100k['sqerr'] = np.square(preds_100k['prediction'] - preds_100k['rating'])
```

```{code-cell} ipython3
sns.catplot(preds_100k, x='algo', y='sqerr', estimator=lambda x: np.sqrt(np.mean(x)), kind='bar')
plt.ylabel('RMSE')
plt.xlabel('Algorithm')
plt.show()
```

## MovieLens 20M

```{code-cell} ipython3
train_20m = pd.read_parquet('data/ml-20m/train.parquet')
test_20m = pd.read_parquet('data/ml-20m/test.parquet')
```

### Bias Model

```{code-cell} ipython3
bias = Bias()
bias.fit(train_20m)
```

```{code-cell} ipython3
bias_preds = batch.predict(bias, test_20m)
bias_preds
```

```{code-cell} ipython3
rmse(bias_preds['prediction'], bias_preds['rating'])
```

### ALSModel

```{code-cell} ipython3
als = BiasedMF(25)
als.fit(train_20m)
```

```{code-cell} ipython3
als_preds = batch.predict(als, test_20m)
als_preds
```

```{code-cell} ipython3
rmse(als_preds['prediction'], als_preds['rating'])
```

### Biased MF

```{code-cell} ipython3
tmf = TorchBiasedMF(25)
tmf.fit(train_20m)
```

```{code-cell} ipython3
tmf_preds = batch.predict(tmf, test_100k)
tmf_preds
```

```{code-cell} ipython3
rmse(tmf_preds['prediction'], tmf_preds['rating'])
```

### Integrate Predictions


```{code-cell} ipython3
preds_20m = pd.concat({
    'Bias': bias_preds,
    'ALS': als_preds,
    'Torch BiasedMF': tmf_preds,
}, names=['algo'])
preds_20m = preds_20m.reset_index('algo').reset_index(drop=True)
preds_20m.head()
```

```{code-cell} ipython3
preds_100k['sqerr'] = np.square(preds_100k['prediction'] - preds_100k['rating'])
```

```{code-cell} ipython3
sns.catplot(preds_100k, x='algo', y='sqerr', estimator=lambda x: np.sqrt(np.mean(x)), kind='bar')
plt.ylabel('RMSE')
plt.xlabel('Algorithm')
plt.show()
```

```{code-cell} ipython3

```
