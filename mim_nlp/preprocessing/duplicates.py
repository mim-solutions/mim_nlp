import re
from itertools import chain, product
from multiprocessing import Pool, cpu_count
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import DisjointSet

from mim_nlp.preprocessing.utils import successive_intervals_boundaries_generator, upper_matrix_boundaries_generator

WHITESPACE = re.compile(r"\s+")


# Jaccard
# Based on https://skeptric.com/jaccard-duplicates/


class Deduplicator:
    def __init__(self):
        self.shingle_weighted = None
        self.shingle_weighted_new_data = None

    def clusterize_partial_duplicates(self, texts, threshold=0.7, n_shingles=3, cpu_workers=cpu_count()):
        """The main method for clusterization of partial duplicates.

        The procedure is the following:

        - Calculate similarity matrix (using jaccard metric).
        - Calculate boolean matrix for which boolean_matrix[i][j] == True
          if and only if text(i) and text(j) are almost duplicates.
        - Use disjoint set datastructure to find all connected sets of texts
          (more precisely these are components in undirected graph which
          adjacency matrix is exactly equal to boolean_matrix).
        """
        unique_texts = list(set(texts))
        self.shingle_weighted = {
            i: self._multiset(self._shingle(self._tokenize(text), n_shingles)) for i, text in enumerate(unique_texts)
        }

        n = len(unique_texts)
        boundaries = upper_matrix_boundaries_generator(n, cpu_workers)
        p = Pool(cpu_workers)
        args = ((b, threshold) for b in boundaries)
        results = p.map(self._create_jaccard_similarity_sparse_matrix, args)
        indexes = list(set(chain(*results)))

        clusters = self._create_clusters_from_indexes(indexes, size=len(unique_texts))
        text_to_cluster = {text: cluster for (text, cluster) in zip(unique_texts, clusters)}

        clusters = [text_to_cluster[text] for text in texts]
        # Here we change order such that indices are numbers 0,1,2,... without any gaps
        clusters_ordered = [clusters.index(x) for x in clusters]

        return clusters_ordered

    def eliminate_partial_duplicates(self, df, text_column, threshold=0.7, n_shingles=3, cpu_workers=cpu_count()):
        """Eliminates rows with almost duplicated text. For consistently annotated texts leaves only one row."""
        # TODO function that doesn't remove duplicates with significant difference in prediction score?
        clusters = self.clusterize_partial_duplicates(df[text_column], threshold, n_shingles, cpu_workers)
        df["cluster_index"] = clusters

        df = df.drop_duplicates(subset="cluster_index", ignore_index=True)
        return df.drop(columns=["cluster_index"])

    def get_score_matrix(self, texts, n_shingles=3):
        unique_texts = list(set(texts))
        self.shingle_weighted = {
            i: self._multiset(self._shingle(self._tokenize(text), n_shingles)) for i, text in enumerate(unique_texts)
        }

        results = self._create_jaccard_similarity_matrix()
        return results

    def _create_jaccard_similarity_matrix(self):
        shingles = self.shingle_weighted
        n = len(shingles)
        scores = np.zeros((n, n))
        indices = ((i, j) for (i, j) in product(range(n), range(n)) if i < j)
        for i, j in indices:
            score = self._fast_jaccard(shingles[i], shingles[j])
            scores[i, j] = score
        return scores

    @staticmethod
    def _create_clusters_from_indexes(indexes, size):
        """The static method uses disjoint set datastructure to find all connected sets of texts.

        More precisely these are components in undirected graph,
        which adjacency matrix is exactly equal to boolean_matrix.
        """
        indices = list(range(size))
        disjoint_set_structure = DisjointSet(indices)

        for idx_pair in indexes:
            disjoint_set_structure.merge(idx_pair[0], idx_pair[1])

        clusters = [disjoint_set_structure[i] for i in indices]
        return clusters

    @staticmethod
    def _tokenize(s: str) -> list:
        """The method splits a string into tokens."""
        return WHITESPACE.split(s)

    @staticmethod
    def _untokenize(ts: Iterable[str]) -> str:
        """The method joins a list of tokens into a string."""
        return " ".join(ts)

    @staticmethod
    def _subseq(seq: list[Any], n: int = 1) -> list[tuple[Any]]:
        """Returns all contiguous subsequences of seq of length n.

        Example: _subseq([1,2,3,4], n=2) == [(1,2), (2,3), (3,4)]
        """
        return [tuple(seq[i : i + n]) for i in range(0, len(seq) + 1 - n)]

    @staticmethod
    def _shingle(seq: list[str], n: int = 1) -> list[str]:
        if len(seq) < n:
            return [Deduplicator._untokenize(seq)]
        return [Deduplicator._untokenize(s) for s in Deduplicator._subseq(seq, n)]

    @staticmethod
    def _multiset(xs):
        seen = {}
        output = set()
        for item in xs:
            if item not in seen:
                seen[item] = 0
            else:
                seen[item] += 1
            output.add((item, seen[item]))
        return output

    # TODO optimize with e.g. numba
    @staticmethod
    def _fast_jaccard(x, y):
        if len(x) == 0 and len(y) == 0:
            return 1
        n = len(x.intersection(y))
        return n / (len(x) + len(y) - n)

    def _create_jaccard_similarity_sparse_matrix(self, args):
        boundaries, threshold = args
        indexes = []
        # copying shingles to local variable reduces processing time but increases memory usage
        shingles = {
            k: v
            for k, v in self.shingle_weighted.items()
            if boundaries[0] <= k < boundaries[1] or boundaries[2] <= k < boundaries[3]
        }
        indices = (
            (i, j)
            for (i, j) in product(range(boundaries[0], boundaries[1]), range(boundaries[2], boundaries[3]))
            if i < j
        )
        for i, j in indices:
            if self._fast_jaccard(shingles[i], shingles[j]) > threshold:
                indexes.append((i, j))
        return indexes

    def eliminate_duplicates_in_new_data(
        self,
        df_old,
        df_new,
        text_column,
        threshold=0.7,
        n_shingles=3,
        cpu_workers=cpu_count(),
    ):
        """The method removes duplicates between two dataframes.

        Both dataframes should contain data without duplicates inside.
        """
        unique_texts_old = list(set(df_old[text_column]))
        unique_texts_new = list(set(df_new[text_column]))

        self.shingle_weighted_new_data = {
            i: self._multiset(self._shingle(self._tokenize(text), n_shingles))
            for i, text in enumerate(unique_texts_old + unique_texts_new)
        }
        boundaries = successive_intervals_boundaries_generator(
            len(unique_texts_old), len(unique_texts_new), n_splits=cpu_workers
        )
        p = Pool(cpu_workers)
        args = ((b, threshold) for b in boundaries)
        results = p.map(self._create_jaccard_similarity_sparse_matrix_new_data, args)
        indexes = list(set(chain(*results)))

        clusters = self._create_clusters_from_indexes(indexes, size=len(unique_texts_old) + len(unique_texts_new))
        text_to_cluster = {text: cluster for (text, cluster) in zip(unique_texts_old + unique_texts_new, clusters)}

        clusters = [text_to_cluster[text] for text in list(df_old[text_column]) + list(df_new[text_column])]
        # Here we change order such that indices are numbers 0,1,2,... without any gaps
        clusters_ordered = [clusters.index(x) for x in clusters]

        df_joined = pd.concat([df_old, df_new], ignore_index=True)
        df_joined["cluster_index"] = clusters_ordered
        df_joined = df_joined.drop_duplicates(subset="cluster_index", ignore_index=True)
        return df_joined.drop(columns=["cluster_index"])

    def _create_jaccard_similarity_sparse_matrix_new_data(self, args):
        boundaries, threshold = args
        indexes = []
        # copying shingles to local variable reduces processing time but increases memory usage
        shingles = {
            k: v
            for k, v in self.shingle_weighted_new_data.items()
            if boundaries[0] <= k < boundaries[1] or boundaries[2] <= k < boundaries[3]
        }
        indices = (
            (i, j) for (i, j) in product(range(boundaries[0], boundaries[1]), range(boundaries[2], boundaries[3]))
        )
        for i, j in indices:
            if self._fast_jaccard(shingles[i], shingles[j]) > threshold:
                indexes.append((i, j))
        return indexes
