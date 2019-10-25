#!/bin/bash

# 5 parts
# train
python shepard_metzler_n_parts.py --n-parts 5 --split train --n_experiments 1000000

# val
python shepard_metzler_n_parts.py --n-parts 5 --split val --n_experiments 50000

# test
python shepard_metzler_n_parts.py --n-parts 5 --split test --n_experiments 50000


# 4 parts
# train
python shepard_metzler_n_parts.py --n-parts 4 --split train --n_experiments 500

# val
python shepard_metzler_n_parts.py --n-parts 4 --split val --n_experiments 500

# test
python shepard_metzler_n_parts.py --n-parts 4 --split test --n_experiments 500


# 6 parts
# train
python shepard_metzler_n_parts.py --n-parts 6 --split train --n_experiments 500

# val
python shepard_metzler_n_parts.py --n-parts 6 --split val --n_experiments 500

# test
python shepard_metzler_n_parts.py --n-parts 6 --split test --n_experiments 500
