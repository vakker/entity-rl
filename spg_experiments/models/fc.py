# pylint: disable=undefined-loop-variable

from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn


class FcPolicy(TorchModelV2, nn.Module):
    # pylint: disable=abstract-method

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(
            obs_space, action_space, num_outputs, model_config, name
        )

    def forward(self, input_dict, state, seq_lens):
        fc_out, state = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, state

    def value_function(self):
        return self.torch_sub_model.value_function()
