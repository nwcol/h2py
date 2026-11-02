## `h2py`: a tool for inferring multi-population history from ancient DNA

This is the repository for `h2py`, a Python package that extends `moments-LD`
(https://github.com/MomentsLD/moments) to study ancient population history
with H<sub>2</sub>.
H<sub>2</sub> is a two-locus genetic statistic, defined as the probability that
two genome copies differ by state at two loci that span some recombination
fraction.

### Contents

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

If you want to edit the code, it's useful to perform an editable install:

```
pip install -e .
```

### Dependencies


### Examples


#### Estimating H<sub>2</sub> from high-coverage data


```
sample1 pop1
sample2 pop2
sample3 pop3
```


```
import h2py

vcf_file = "my/path/input.vcf"
pop_file = "my/path/pops.txt"
rec_file = "my/path/rec_map.tsv"
bed_file = "my/bath/filter.bed"

sums = h2py.parsing.compute_h2_stats(
    vcf_file=vcf_file, pop_file=pop_file, bed_file=bed_file,
    rec_map_file=rec_file, rec_pos_col="Position", rec_map_col="Map"
)
```

Here, `sums` is a dictionary with these keys:

- `"pops"`: A list of population labels, derived from `pop_file`.

- `"stats"`: A tuple of labels for H<sub>2</sub> and H statisitcs. These have
the form `"H2_{i}_{j}"`, where `i` and `j` index population labels..

- `"bins"`: A list of tuples, each defining the lower and upper bound
of a bin.

- `"sums"`: Raw sums of H<sub>2</sub> statistics listed in `"stats"`. The last
element is an array of one-locus diversity statistics.

- `"denoms"`: Denominators for normalizing H<sub>2</sub> and H statistics; one
per bin.

To compute means and bootstrap covariances across many intervals, we can use:

```
intervals = [[5_000_000 * i, 5_000_000 * (i + 1)] for i in range(20)]
all_sums = {
    i: h2py.parsing.compute_h2_stats(
        vcf_file=vcf_file, pop_file=pop_file, bed_file=bed_file,
        rec_map_file=rec_file, rec_pos_col="Position", rec_map_col="Map",
        interval=intervals[i]
    ) for i in range(len(intervals))}
means, varcovs = h2py.parsing.bootstrap_data(all_sums)
```

#### Inferring demographic parameters


#### Quantifying parameter uncertainty


#### Performing model choice


### Citing the package
