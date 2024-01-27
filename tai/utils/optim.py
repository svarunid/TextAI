from optax import adam, warmup_cosine_decay_schedule


def optimizer(config):
    """
    Create an optimizer from a config dictionary.
    """
    lr = warmup_cosine_decay_schedule(
        config["lr"],
        config["peak_lr"],
        config["warmup_steps"],
        config["decay_steps"],
        config["final_lr"],
    )
    optimizer = adam(lr)
    return optimizer
