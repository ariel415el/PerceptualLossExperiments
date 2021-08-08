
class config:
    def __init__(self):
        pass

    def save(self, path):
        f = open(path, 'w')
        f.write(f"img_dim={self.img_dim}\n")
        f.write(f"z_dim={self.z_dim}\n")
        f.write(f"lr={self.lr}\n")
        f.write(f"decay_epochs={self.decay_epochs}\n")
        f.write(f"decay_rate={self.decay_rate}\n")
        f.write(f"decay_rate={self.decay_rate}\n")
        f.write(f"num_epochs={self.num_epochs}\n")
        f.close()

default_config = config()
default_config.img_dim = 128
default_config.channels = 3
default_config.z_dim = 256

default_config.lr = 0.001
default_config.decay_epochs = 25
default_config.decay_rate = 0.75
default_config.batch_size = 64
default_config.num_epochs = 200

# GLO parameters
# default_config.e_dim = 64
# default_config.generator_lr_factor = 0.1
# default_config.num_z_steps = 1
# default_config.force_norm = True