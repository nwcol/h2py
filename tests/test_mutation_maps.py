
import numpy as np
import pandas
import pytest

import h2py


class TestVectorLoading:

    def test_basic_case(self):
        filename = "data/simple_mutation_map.npy"
        vec = np.array([1e-8, 1e-7, 1e-8, 1.5e-8, 1.2e-9])
        np.save(filename, vec)
        loaded_vec = h2py.utils._read_mut_map_file(filename)
        assert np.all(loaded_vec == vec)

    def test_length_adjustments(self):
        filename = "data/simple_mutation_map.npy"
        vec = np.array([1e-8, 1e-7, 1e-8, 1.5e-8, 1.2e-9])
        np.save(filename, vec)
        loaded_vec = h2py.utils._read_mut_map_file(filename, L=4)
        expected = vec[:4]
        assert np.all(loaded_vec == expected)

        loaded_vec = h2py.utils._read_mut_map_file(filename, L=10)
        expected = np.concatenate([vec, [np.nan] * 5])
        print(loaded_vec, expected)
        assert len(loaded_vec) == len(expected)
        assert np.all(loaded_vec[:5] == expected[:5])


class TestTableLoading:

    def test_tsv_case(self):
        filename = "data/mutation_map.tsv"
        df = pandas.DataFrame(
            {"chromStart": [0, 10, 20],
            "chromEnd": [10, 20, 30],
            "mutRate": [1e-8, 1.11e-8, 1.09e-8]}
        )
        df.to_csv(filename, sep="\t")
        loaded_map = h2py.utils._read_mut_map_file(filename)
        expected = np.concatenate([[1e-8] * 10, [1.11e-8] * 10, [1.09e-8] * 10])
        assert np.all(loaded_map == expected)

    def test_csv_case(self):
        filename = "data/mutation_map.csv"
        df = pandas.DataFrame(
            {"chromStart": [0, 10, 20],
             "chromEnd": [10, 20, 30],
             "mutRate": [1e-8, 1.11e-8, 1.09e-8]}
        )
        df.to_csv(filename)
        loaded_map = h2py.utils._read_mut_map_file(filename)
        expected = np.concatenate([[1e-8] * 10, [1.11e-8] * 10, [1.09e-8] * 10])
        assert np.all(loaded_map == expected)

    def test_missing_data(self):
        filename = "data/missing_data_mutation_map.csv"
        df = pandas.DataFrame(
            {"chromStart": [0, 10, 20],
            "chromEnd": [10, 20, 30],
            "mutRate": [1e-8, np.nan, 1.09e-8]}
        )
        df.to_csv(filename)
        loaded_map = h2py.utils._read_mut_map_file(filename)
        expected = np.concatenate([[1e-8] * 10, [np.nan] * 10, [1.09e-8] * 10])
        assert len(loaded_map) == 30
        assert np.all(np.isnan(loaded_map[10:20]))

    def test_implicit_missing_data(self):
        filename = "data/gap_mutation_map.csv"
        df = pandas.DataFrame(
            {"chromStart": [0, 20],
             "chromEnd": [10, 30],
             "mutRate": [1e-8, 1.09e-8]}
        )
        df.to_csv(filename)
        loaded_map = h2py.utils._read_mut_map_file(filename)
        assert len(loaded_map) == 30
        assert np.all(np.isnan(loaded_map[10:20]))

