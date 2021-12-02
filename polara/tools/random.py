import numpy as np

def check_random_state(random_state):
    '''
    Handles seed or random state as an input. Provides consistent output.
    '''
    if random_state is None:
        return np.random
    if isinstance(random_state, (np.integer, int)):
        return np.random.RandomState(random_state)
    return random_state

def random_seeds(size, entropy=None):
    '''Generates a sequence of most likely independent seeds.'''
    return np.random.SeedSequence(entropy).generate_state(size)

def seed_generator(seed):
    rs = np.random.RandomState(seed)
    max_int = np.iinfo('i4').max
    while True:
        new_seed = yield rs.randint(max_int)
        if new_seed is not None:
            rs = np.random.RandomState(new_seed)
