## `h2py`: a tool for inferring multi-population history from ancient DNA

This is `h2py`, a Python package for evolutionary inference with sparse
sampling.
`h2py` is an extension of `moments.LD` (https://github.com/MomentsLD/moments)
built around H<sub>2</sub>, a two-locus summary statistic.
See the documentation for more details, examples, and the API.

### Installation
You can install `h2py` directly from github using `pip`:

```
pip install git+https://github.com/nwcol/h2py.git
```

Or you can clone this repository and install the package locally:

```
git clone https://github.com/nwcol/h2py.git
cd h2py
pip install -r requirements.txt
pip install .
```

Dependencies of `h2py` include `moments`, `demes`, `numpy`, and `scipy`.
To run tests, you will also need to install `pytest`.
If you want to edit the code, it's useful to perform an editable install:

```
pip install -e .
```

### Changelog

Version history is recorded in the documentation.

### Citing the package

If you use `h2py` in your research, please cite

Collier, N. W., Gravel, S., and Ragsdale, A. P. (2026) 
Inferring hominin history with recurrent gene flow from single unphased genomes and a two-locus statistic.
BioRxiv, doi: [https://doi.org/10.64898/2026.04.11.717825](https://doi.org/10.64898/2026.04.11.717825)

```
@article{collier2026,
  title={Inferring hominin history with recurrent gene flow from single unphased genomes and a two-locus statistic},
  author={Collier, Nicholas W and Gravel, Simon and Ragsdale, Aaron P},
  journal={bioRxiv},
  year={2026},
  doi={10.64898/2026.04.11.717825}
}
```

