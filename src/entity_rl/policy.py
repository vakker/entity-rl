# pylint: disable=abstract-method

import json

import torch
from ray.rllib.algorithms.ppo import PPO as PPOTrainer
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.policy.torch_policy import _directStepOptimizerSingleton
from ray.rllib.utils import force_list
from torch.cuda.amp import GradScaler


class PPOTorchPolicyAMP(PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        config["amp"] = config.get("amp", False)
        config["model"]["amp"] = config["amp"]

        # Scalers are initialized in optimizer()
        self._scalers = None
        super().__init__(observation_space, action_space, config)

        assert self.model.use_amp == config["amp"]

        if config.get("show_model"):
            print("################# Model parameters #################")
            print(json.dumps(self.model.num_params, indent=4))

    def loss(self, *args, **kwargs):
        losses = force_list(super().loss(*args, **kwargs))

        assert len(losses) == len(self._scalers)
        assert len(losses) == 1

        scaled_losses = []
        for loss, scaler in zip(losses, self._scalers):
            scaled_losses.append(scaler.scale(loss))

        return scaled_losses

    def extra_grad_process(self, local_optimizer, loss):
        assert len(self._scalers) == 1
        self._scalers[0].unscale_(local_optimizer)

        return super().extra_grad_process(local_optimizer, loss)

    def stats_fn(self, train_batch):
        stats = super().stats_fn(train_batch)
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        if grads:
            grads = [torch.linalg.vector_norm(g) ** 2 for g in grads]
            stats["grad_norm"] = torch.sqrt(sum(grads)).item()
        else:
            stats["grad_norm"] = None

        return stats

    def optimizer(self):
        optimizers = super().optimizer()
        self._scalers = [GradScaler(enabled=self.config["amp"]) for _ in optimizers]
        return optimizers

    def apply_gradients(self, gradients) -> None:
        if gradients == _directStepOptimizerSingleton:
            for opt, scaler in zip(self._optimizers, self._scalers):
                scaler.step(opt)
                scaler.update()
        else:
            raise NotImplementedError("apply_gradients")
            # # TODO(sven): Not supported for multiple optimizers yet.
            # assert len(self._optimizers) == 1
            # for g, p in zip(gradients, self.model.parameters()):
            #     if g is not None:
            #         if torch.is_tensor(g):
            #             p.grad = g.to(self.device)
            #         else:
            #             p.grad = torch.from_numpy(g).to(self.device)

            # self._optimizers[0].step()


# pylint: disable=abstract-method,arguments-differ,arguments-differ
class PPOTrainerAMP(PPOTrainer):
    _allow_unknown_configs = True

    # @classmethod
    def get_default_policy_class(self, config):
        return PPOTorchPolicyAMP
