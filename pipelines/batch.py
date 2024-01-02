from timeit import default_timer as timer


def time_per_step(size, step, *inputs):
    start = timer()
    step(*inputs)
    end = timer()
    time = end - start
    throughput = size / time
    return throughput, time
