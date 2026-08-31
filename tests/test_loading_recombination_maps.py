
import io
import numpy as np
import pytest

import h2py


class TestColumnFormat:

    def test_generic_hapmap(self):
        data = io.StringIO("""\
Chromosome\tPosition(bp)\tRate(cM/Mb)\tMap(cM)
chr1\t1\t1\t0
chr1\t10\t0\t0.2
chr1\t20\t0\t0.4
chr1\t30\t0\t0.4
""")
        # Note that `Rate` is ignored.
        pos, coords = h2py.utils._read_rec_map_file(data)
        assert np.all(pos == np.array([1, 10, 20, 30]))
        # Map(cM) is converted to Morgans
        assert np.all(coords == np.array([0, 0.2, 0.4, 0.4]) / 100)

    def test_alternate_columns(self):
        data = io.StringIO("""\
chrom\tpos\tmap
chr1\t1\t0
chr1\t10\t0.2
chr1\t20\t0.4
chr1\t30\t0.4
""")
        pos, coords = h2py.utils._read_rec_map_file(
            data, pos_col="pos", map_col="map")
        assert np.all(pos == np.array([1, 10, 20, 30]))
        assert np.all(coords == np.array([0, 0.2, 0.4, 0.4]) / 100)


class TestSeparators:

    def test_comma_separated_file(self):
        data = io.StringIO("""\
Chromosome,Position(bp),Rate(cM/Mb),Map(cM)
chr1,1,1,0
chr1,10,0,0.2
chr1,20,0,0.4
chr1,30,0,0.4
""")
        pos, coords = h2py.utils._read_rec_map_file(data)
        assert np.all(pos == np.array([1, 10, 20, 30]))
        assert np.all(coords == np.array([0, 0.2, 0.4, 0.4]) / 100)

