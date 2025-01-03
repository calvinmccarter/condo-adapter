# condo-adapter

[![PyPI version](https://badge.fury.io/py/condo.svg)](https://badge.fury.io/py/condo.svg)
[![Downloads](https://static.pepy.tech/badge/condo/month)](https://static.pepy.tech/badge/condo/month)
[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

ConDo Adapter performs Confounded Domain Adaptation, which corrects for
batch effects while conditioning on confounding variables.
We hope it sparks joy as you clean up your data!

## Installation

### Installation from pip

You can install the toolbox through PyPI with:

```console
pip install condo
```

Note: If you have issues with importing `torchmin`, you may need to install from source, as shown below. Or you can try re-installing [pytorch-minimize](https://github.com/rfeinman/pytorch-minimize) from source. 

### Installation from source

After cloning this repo, install the dependencies on the command-line via:

```console
pip install -r requirements.txt
```

In this directory, run

```console
pip install -e .
```

## Usage

Import ConDo and create the adapter:
```python
from condo import ConDoAdapterKLD
condoer = ConDoAdapterKLD()
```

Try using it:
```python
import numpy as np

X_T = np.sort(np.random.uniform(0, 8, size=(100, 1)))
X_S = np.sort(np.random.uniform(4, 8, size=(100, 1)))
Y_T = np.random.normal(4 * X_T + 1, 1 * X_T + 1)
Y_Strue = np.random.normal(4 * X_S + 1, 1 * X_S + 1)
Y_S = 5 * Y_Strue + 2
condoer.fit(Y_S, Y_T, X_S, X_T)
Y_S2T = condoer.transform(Y_S)
print(f"before ConDo: {np.mean((Y_S - Y_Strue) ** 2):.3f}")
print(f"after ConDo:  {np.mean((Y_S2T - Y_Strue) ** 2):.3f}")
```

More thorough examples are provided in the `examples` directory, and all code for experiments are in the `papers` directory.

## Development

### Testing
In this directory run
```console
pytest
```
### Citing this toolbox

If you use this toolbox in your research and find it useful, please cite ConDo
using the following reference to the [TMLR paper](https://openreview.net/pdf?id=GSp2WC7q0r):

In Bibtex format:

```bibtex
@article{
mccarter2024towards,
title={Towards Backwards-Compatible Data with Confounded Domain Adaptation},
author={Calvin McCarter},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024},
url={https://openreview.net/forum?id=GSp2WC7q0r},
note={}
}
```

### License Information

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg


