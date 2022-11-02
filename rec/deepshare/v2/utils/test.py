#!/usr/bin/env.txt python
# coding: utf-8

import numpy as np

if __name__ == '__main__':
    s = [[285371, 0.4349], [144978, 0.4349], [96613, 0.4349], [96618, 0]]
    print(list(filter(lambda x: x[1] > 0.0, s)))
