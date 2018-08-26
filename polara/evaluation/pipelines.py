from __future__ import print_function

from operator import mul as mul_op
from functools import reduce
from itertools import product
from random import choice


def random_chooser():
    while True:
        values = yield
        yield choice(values)


def random_grid(params, n=60, grid_cache=None):
    if not isinstance(n, int):
        raise TypeError('n must be an integer, not {}'.format(type(n)))
    if n < 0:
        raise ValueError('n should be >= 0')

    grid = grid_cache or set()
    max_n = reduce(mul_op, [len(vals) for vals in params.values()])
    n = min(n if n > 0 else max_n, max_n)
    param_chooser = random_chooser()
    try:
        while len(grid) < n:
            level_choice = []
            for v in params.values():
                next(param_chooser)
                level_choice.append(param_chooser.send(v))
            grid.add(tuple(level_choice))
    except KeyboardInterrupt:
        print('Interrupted by user. Providing current results.')
    return grid


def set_config(model, attributes, values):
    for name, value in zip(attributes, values):
        setattr(model, name, value)
