from timeit import default_timer as timer


class Timer:
    def __init__(self, model_name='Model', verbose=True, msg=None):
        self.model_name = model_name
        self.message = msg or '{} training time: {}s'
        self.elapsed_time = []
        self.verbose = verbose

    def __enter__(self):
        self.start = timer()
        return self.elapsed_time

    def __exit__(self, type, value, traceback):
        self.elapsed_time.append(timer() - self.start)
        if self.verbose:
            print(self.message.format(self.model_name, self.elapsed_time[-1]))
