
class config:
    def __init__(self):
        pass

default_config = config()
default_config.img_dim = 128
default_config.channels = 3
default_config.z_dim = 64
default_config.e_dim = 64

default_config.lr = 0.001
default_config.decay_epochs = 25
default_config.decay_rate = 0.8
default_config.generator_lr_factor = 0.1
default_config.batch_size = 64
default_config.num_z_steps = 1
default_config.num_epochs = 100
default_config.force_norm = True