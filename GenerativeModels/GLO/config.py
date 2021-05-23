
class config:
    def __init__(self):
        pass

faces_config = config()
faces_config.img_dim = 128
faces_config.channels = 3
faces_config.z_dim = 64
faces_config.e_dim = 64
faces_config.use_bn = False

faces_config.lr = 0.001
faces_config.decay_epochs = 25
faces_config.decay_rate = 0.8
faces_config.generator_lr_factor = 1
faces_config.batch_size = 64
faces_config.num_z_steps = 1
faces_config.num_epochs = 200
faces_config.force_norm = True