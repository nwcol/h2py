"""
Test missing data handling and its interaction with masking in
GenotypeProbMatrix.
"""

import numpy as np
import os
import tempfile
import pytest

import h2py
from h2py import GenotypeProbMatrix, GeneticMask


class TestMissingDataMask:

    def test_basic_missing_data_construction(self):
        # Example genotype probs with missing data
        data = np.array(
            [[-1, -1, -1, 0, 1, 0],
             [0, 1, 0, 0, 1, 0],
             [0, 1, 0, -1, -1, -1]], dtype=np.float64
            )
        expected = np.array([[False, True], [True, True], [True, False]])
        non_missing = GenotypeProbMatrix.find_non_missing_data(data)
        assert np.all(expected == non_missing)

    def test_partial_missing_data_construction(self):
        data = np.array(
            [[-1, 0, 0, 0, 1, 0],
             [0, 1, 0, 0, 1, 0],
             [0, 1, 0, 0, -1, 0]], dtype=np.float64)
        expected = np.array([[False, True], [True, True], [True, False]])
        non_missing = GenotypeProbMatrix.find_non_missing_data(data)
        assert np.all(expected == non_missing)


class TestSampleSlicing:

    def test_single_sample_access(self):
        data = np.array(
            [[0.99, 0.01, 0.0, 0.3, 0.4, 0.3, 0.15, 0.85, 0.0],
             [-1., -1., -1., 0.15, 0.7, 0.15, 0.8, 0.2, 0.0],
             [0, 0.5, 0.5, -1, -1, -1, -1, -1, -1],
             [0.1, 0.9, 0, 0.9, 0.1, 0, -1, -1, -1]]
        )
        sites = np.arange(4)
        matrix = GenotypeProbMatrix(data, sites, {"pop0": [0, 1, 2]})
        expected = matrix.genotype_probs[np.ix_([0, 2, 3], [0, 1, 2])]
        result = matrix.slice_sample(0)
        assert np.all(result == expected)
        expected = matrix.genotype_probs[np.ix_([0, 1, 3], [3, 4, 5])]
        result = matrix.slice_sample(1)
        assert np.all(result == expected)
        expected = matrix.genotype_probs[np.ix_([0, 1], [6, 7, 8])]
        result = matrix.slice_sample(2)
        assert np.all(result == expected)

    def test_two_sample_access(self):
        data = np.array(
            [[0.99, 0.01, 0.0, 0.3, 0.4, 0.3, 0.15, 0.85, 0.0],
             [-1., -1., -1., 0.15, 0.7, 0.15, 0.8, 0.2, 0.0],
             [0, 0.5, 0.5, -1, -1, -1, -1, -1, -1],
             [0.1, 0.9, 0, 0.9, 0.1, 0, -1, -1, -1]],
        )
        sites = np.arange(4)
        matrix = GenotypeProbMatrix(data, sites, {"pop0": [0, 1, 2]})
        expected = (matrix.genotype_probs[np.ix_([0, 3], [0, 1, 2])],
                    matrix.genotype_probs[np.ix_([0, 3], [3, 4, 5])])
        result = matrix.slice_samples([0, 1])
        assert np.all(result[0] == expected[0])
        assert np.all(result[1] == expected[1])



