## `h2py`: a tool for inferring multi-population history from ancient DNA

This is `h2py`, a Python package for evolutionary inference with sparse
sampling.
`h2py` is an extension of [`moments.LD`](https://github.com/MomentsLD/moments)
built around H<sub>2</sub>, a two-locus summary statistic.
See the documentation for more details, examples, and the API.

### Installation
We can install `h2py` directly from github using `pip`:

```
pip install git+https://github.com/nwcol/h2py.git
```

Or we can clone this repository and install the package locally, then install
the required dependencies.

```
git clone https://github.com/nwcol/h2py.git
cd h2py
pip install -r requirements.txt
pip install .
```

Dependencies of `h2py` include `moments`, `demes`, `numpy`, and `scipy`.
To run tests, we will also need to install `pytest`.
To edit the code, it's useful to perform an editable install:

```
pip install -e .
```

### Contents

Typical usage for `h2py` might involve estimating H<sub>2</sub> from sequence
data stored in a VCF file, fitting parameters for several demographic models,
estimating parameter confidence intervals, and making formal comparisons of
goodness-of-fit between competing models.
The package currently includes functionality to:

- Estimate H<sub>2</sub> with specified filtering, masking, phasing,
  recombination and mutation maps.

- Bootstrap estimated statistics across genomic windows to estimate
  covariances for the likelihood function

- Estimate parameters for demographic models using maximum likelihood.

- Estimate parameter uncertainty with several distinct methods.

- Plot estimated statistics and model expectations.

Planned elements include:

- Estimation of H<sub>2</sub> from genotype probabilities for lower-coverage
  sequences.

- Numerical computation of the variance of H<sub>2</sub> using machinery from
  `moments.LD`.

- Joint inference using H<sub>2</sub> alongside the site frequency spectrum
  (SFS) or LD statistics from `moments.LD`.

- Monte-Carlo two-locus genealogical simulations.

This repository holds [examples](examples) that illustrate usage with
simulated data and some generic command-line [scripts](scripts).

### Changelog

Version history is recorded in the documentation.

### Citing the package

If you use `h2py` in your research, please cite

Collier, N. W., Gravel, S., and Ragsdale, A. P. (2026) 
Inferring hominin history with recurrent gene flow from single unphased
genomes and a two-locus statistic.
BioRxiv, doi: [https://doi.org/10.64898/2026.04.11.717825](https://doi.org/10.64898/2026.04.11.717825)

```
@article{collier2026,
  title={Inferring hominin history with recurrent gene flow from single
         unphased genomes and a two-locus statistic},
  author={Collier, Nicholas W and Gravel, Simon and Ragsdale, Aaron P},
  journal={bioRxiv},
  year={2026},
  doi={10.64898/2026.04.11.717825}
}
```

