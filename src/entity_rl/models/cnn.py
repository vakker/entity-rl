from ray.rllib.models.torch.visionnet import VisionNetwork

# from .base import count_parameters


class CnnPolicy(VisionNetwork):
    # pylint: disable=abstract-method,useless-super-delegation

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # print("#################")
        # print(self)
        # print(count_parameters(self))
