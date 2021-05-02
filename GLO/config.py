
class config:
    def __init__(self):
        pass

mnist_configs = config()
mnist_configs.img_dim = 28
mnist_configs.channels = 1
mnist_configs.z_dim = 64
mnist_configs.e_dim = 32
mnist_configs.use_bn = True

mnist_configs.lr = 0.01
mnist_configs.generator_lr_factor = 0.1
mnist_configs.batch_size = 128
mnist_configs.decay_epochs = 20
mnist_configs.decay_rate = 0.5
mnist_configs.num_epochs = 200
mnist_configs.force_norm = True

faces_config = config()
faces_config.img_dim = 64
faces_config.channels = 3
faces_config.z_dim = 64
faces_config.e_dim = 64
faces_config.use_bn = False

faces_config.lr = 0.01
faces_config.decay_epochs = 10
faces_config.decay_rate = 0.95
# faces_config.lr = 0.02
# faces_config.decay_epochs = 50
# faces_config.decay_rate = 0.9
faces_config.generator_lr_factor = 0.1
faces_config.batch_size = 256
faces_config.num_epochs = 100
faces_config.num_opt_steps = 3
faces_config.force_norm = False