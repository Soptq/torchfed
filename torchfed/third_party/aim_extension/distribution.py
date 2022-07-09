from aim.sdk.objects.distribution import Distribution as OriginDistribution
import numpy as np


class Distribution(OriginDistribution):
    def __init__(self, distribution, bin_count=64):
        super().__init__(distribution, bin_count)
        # convert to np.histogram
        try:
            np_histogram = np.histogram(
                distribution, bins=bin_count, range=(-0.5, bin_count - 0.5))
        except TypeError:
            raise TypeError(
                f'Cannot convert to aim.Distribution. Unsupported type {type(distribution)}.')
        self._from_np_histogram(np_histogram)
