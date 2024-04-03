from __future__ import annotations

import math
from collections.abc import Generator
from itertools import product


def upper_matrix_boundaries_generator(
    matrix_len: int, n_splits: int
) -> Generator[tuple[int, int, int, int], None, None]:
    split_len = matrix_len / n_splits
    for i, j in product(range(n_splits), range(n_splits)):
        if i < j:
            # min_x, max_x, min_y, max_y
            yield math.floor(i * split_len), math.ceil((i + 1) * split_len), math.floor(j * split_len), math.ceil(
                (j + 1) * split_len
            )

    for i in range(n_splits):
        yield math.floor(i * split_len), math.ceil((i + 1) * split_len), math.floor(i * split_len), math.ceil(
            (i + 1) * split_len
        )


def successive_intervals_boundaries_generator(
    first_interval_length: int, second_interval_length: int, n_splits: int
) -> Generator[tuple[int, int, int, int], None, None]:
    # split longer dimension of rectangle
    if second_interval_length > first_interval_length:
        for x in successive_intervals_boundaries_generator(second_interval_length, first_interval_length, n_splits):
            yield x[2], x[3], x[0], x[1]

    split_len = first_interval_length / n_splits
    for i in range(n_splits):
        # min_x, max_x, min_y, max_y
        yield math.floor(i * split_len), math.ceil(
            (i + 1) * split_len
        ), first_interval_length, first_interval_length + second_interval_length
