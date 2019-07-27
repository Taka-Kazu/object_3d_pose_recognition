#!/usr/bin/env python
#! coding utf-8

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import numpy as np
import scipy.spatial as ss
from pprint import pprint
import time

from euclidean_clustering import EuclideanClustering

if __name__=='__main__':
    pc = 3 * (np.random.rand(2000, 3) - 0.5)

    ec = EuclideanClustering()
    ec.set_params(1, 3, 6, 10)
    start = time.time()
    clusters = ec.calculate(pc)
    print(time.time() - start)
    pprint(("cluster num: ", len(clusters)))
