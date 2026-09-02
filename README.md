# Exact Haar Statistics of Planar \(k\)-Purity

Code associated with the paper

**Kourosh Mirsohi, “Exact Haar Statistics of Planar \(k\)-Purity in Multipartite Quantum Systems.”**  
arXiv:2608.28914 (2026).

- Paper: https://arxiv.org/abs/2608.28914
- PDF: https://arxiv.org/pdf/2608.28914

This repository contains the analysis and figure-generation code used to study the Haar statistics of planar \(k\)-purity for multipartite quantum systems. The implementation includes exact first- and second-moment formulas for arbitrary local dimension, balanced-case higher moments, Haar-random-state simulations, regression tests, and scripts for the contraction diagrams used in the manuscript.

## Repository contents

| File | Description |
| --- | --- |
| `purity_analysis.py` | Exact formulas, Haar-state simulations, statistical summaries, and manuscript figure generation |
| `test_purity_analysis.py` | Regression tests for the analytic formulas and numerical implementation |
| `delta_coupling_figures.py` | Generates the representative Kronecker-delta contraction diagrams |
| `original_notebook.ipynb` | Original exploratory notebook used during development |
| `requirements.txt` | Python dependencies |

The complete reproducibility archive associated with the arXiv submission additionally contains the saved Monte Carlo datasets, generated results, figures, checksums, and supporting reports. Those data files are not duplicated in this code-only GitHub repository.

## Requirements

Python **3.10 or newer** is recommended.

The main dependencies are:

- NumPy
- SciPy
- Matplotlib

Create a virtual environment and install the dependencies with

```bash
git clone https://github.com/Mirsohi/Haar-statistics.git
cd Haar-statistics

python -m venv .venv
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Then install the requirements:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick start: exact formulas

The exact analytic functions do not require any saved simulation data. For example:

```python
import purity_analysis as pa

# Haar mean and variance of planar k-purity
n = 10
p = 2
k = 5

mean = pa.exact_planar_mean_local_dimension(n, p, block_size=k)
variance = pa.exact_planar_variance_local_dimension(n, p, block_size=k)

print("mean =", mean)
print("variance =", variance)

# Balanced-qubit skewness
print("skewness =", pa.exact_planar_skewness(n))
```

The code also exposes the overlap-dependent pair covariance, arbitrary-\(k\) second moments, balanced absolute-purity variance, and the permutation-based higher-moment routines used in the paper.

## Haar-random-state simulation

Small simulations can be run directly from the public code without the saved manuscript datasets:

```python
import numpy as np
import purity_analysis as pa

n = 6
p = 2
k = 3
samples = 1000

rng = np.random.default_rng(12345)
states = pa.haar_state_batch(
    n=n,
    batch_size=samples,
    rng=rng,
    local_dimension=p,
)

subsystems = pa.planar_subsystems(
    n=n,
    unique=True,
    block_size=k,
)

values = pa.averaged_purity_batch(
    states,
    n=n,
    subsystems=subsystems,
    local_dimension=p,
)

print("sample mean:", values.mean())
print("exact mean:", pa.exact_planar_mean_local_dimension(n, p, k))
```

The manuscript simulations use fixed seeds and batched sampling so that the reported validation datasets can be reproduced exactly from the full reproducibility archive.

## Reproducing the manuscript results

The full reproducibility package accompanying the arXiv submission uses the directory layout

```text
project/
├── code/
│   ├── purity_analysis.py
│   ├── test_purity_analysis.py
│   └── delta_coupling_figures.py
├── data/
├── results/
└── figures/
```

`purity_analysis.py` was written against this archive layout. In the full archive, the main commands are:

```bash
python code/purity_analysis.py summarize
python code/purity_analysis.py simulate-comparisons --samples 40000
python code/purity_analysis.py figures
```

The exact publication-validation datasets use fixed random seeds recorded by the script and the accompanying archive metadata.

### Regression tests

With the full archive layout and saved datasets present, run

```bash
python -m unittest code/test_purity_analysis.py
```

The regression suite checks, among other things:

- the planar-subsystem convention;
- exact Haar means and variances;
- even/odd balanced closed forms;
- arbitrary-\(k\) and qudit formulas;
- direct \(S_4\) permutation enumeration;
- the representative contraction counts;
- recovery of planar and absolute-purity variances from the pair kernel;
- third-moment/skewness formulas;
- physicality of sampled purities;
- asymptotic coefficients; and
- the expected sizes of the saved validation datasets.

## Contraction diagrams

`delta_coupling_figures.py` generates the vector diagrams illustrating two representative four-replica Kronecker-delta contractions used in the derivation.

In the full archive layout, run

```bash
python code/delta_coupling_figures.py
```

The resulting PDF and PNG files are written to the archive's `figures/` directory.

## Reproducibility notes

- Haar-random pure states are sampled as normalized complex Gaussian vectors.
- Simulations are performed in batches to avoid constructing unnecessary full density matrices.
- The publication-validation runs use fixed seeds.
- Exact mean and variance formulas support arbitrary local dimension \(p\) and every
  \(1 \le k \le \lfloor n/2 \rfloor\).
- The balanced-qubit implementation includes exact finite-sum evaluation of the third moment and standardized skewness.
- Manuscript figures are generated as vector PDFs in addition to raster previews.

For the exact data files used in the paper, use the reproducibility archive accompanying arXiv:2608.28914.

## Citation

If you use this code or the associated results, please cite:

```bibtex
@article{mirsohi2026haar,
  title   = {Exact Haar Statistics of Planar $k$-Purity in Multipartite Quantum Systems},
  author  = {Mirsohi, Kourosh},
  year    = {2026},
  eprint  = {2608.28914},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  doi     = {10.48550/arXiv.2608.28914}
}
```

## License

No software license is currently specified for this repository. If you intend others to reuse, modify, or redistribute the code, add an explicit open-source license (for example, MIT or BSD-3-Clause).
