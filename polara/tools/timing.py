from contextlib import contextmanager
from timeit import default_timer as timer
from string import Template


training_time_message = Template('$model training time: ${time}')


def format_elapsed_time(seconds_elapsed):
    minutes, seconds = divmod(seconds_elapsed, 60)
    hours, minutes = divmod(minutes, 60)

    if hours == 0:
        if minutes == 0:
            return f'{seconds:.3f}s'
        return f'{minutes:>02.0f}m:{seconds:>02.0f}s'
    return f'{hours:.0f}h:{minutes:>02.0f}m:{seconds:>02.0f}s'


@contextmanager
def track_time(time_container=None, verbose=False, message=None, **kwargs):
    if time_container is None:
        time_container = []
    start = timer()
    try:
        yield time_container
    finally:
        stop = timer()
        elapsed = stop - start
        time_container.append(elapsed)
    if verbose:
        message = message or training_time_message
        elapsed_time = format_elapsed_time(elapsed)
        print(message.safe_substitute(kwargs, time=elapsed_time))
