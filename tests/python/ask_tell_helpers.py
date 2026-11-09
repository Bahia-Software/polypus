"""A deterministic ask/tell optimizer for the ``polypus.External`` tests.

``RandomSearch`` draws its candidates with its own ``random.Random``, so two
instances built with the same arguments propose the same batches whatever the
values they are told: the tests can compare what the bridge evaluated and told
with what was asked, and the C-7 tests can vary Polypus's seed alone.
"""

import math
import random


class RandomSearch:
    """Uniform random candidates in ``[low, high)^dimensions``.

    The batch sizes cycle through ``sizes``. After ``batches`` batches (never
    with ``None``), ``ask()`` returns ``end``: ``[]`` or ``None``, the two ways
    to say the search is done. ``asked`` keeps every batch ``ask()`` returned
    and ``told`` every ``(candidates, values)`` pair ``tell`` received.
    """

    def __init__(
        self,
        dimensions,
        sizes=(4,),
        *,
        seed=0,
        batches=None,
        end=(),
        low=0.0,
        high=2 * math.pi,
    ):
        self.dimensions = dimensions
        self.sizes = list(sizes)
        self.random = random.Random(seed)
        self.batches = batches
        self.end = list(end) if end is not None else None
        self.low = low
        self.high = high
        self.asked = []
        self.told = []

    def ask(self):
        if self.batches is not None and len(self.asked) >= self.batches:
            return self.end
        size = self.sizes[len(self.asked) % len(self.sizes)]
        batch = [
            [self.random.uniform(self.low, self.high) for _ in range(self.dimensions)]
            for _ in range(size)
        ]
        self.asked.append(batch)
        return batch

    def tell(self, candidates, values):
        self.told.append((candidates, values))
