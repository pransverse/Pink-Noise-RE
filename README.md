# Pink-Noise-RE

(For reproducing, please fork the repository from [stable_baselines3](https://github.com/DLR-RM/stable-baselines3) and make the above changes.)

## Contents
The repo is divided into 5 sections 
- SAC implementation of coloured noise on 9 environments and with 10 beta values
- MPO implementation of coloured noise on 9 environments
- TD3 implementation of coloured noise on 9 environments
- SAC implementation of colour-schedulers: linear, cosine and atanh as decay functions 
- SAC and TD3 implementation of spatio-temporal noise

## Requirements
We have utilised a modified version of stable_baselines3 with the following changes, with the purpose of passing observation (**obs**) to the final sample function so that spatio-temporal noise can be added based on the state visitation count: 
stable_baselines3/common/distributions.py, line 89: 
        return self.sample() -> _return self.sample(**obs**)_
stable_baselines3/common/distributions.py, all log_prob_from_params function definitions:
        def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor) -> Tuple[th.Tensor, th.Tensor]: ->
        _def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor, **obs**) -> Tuple[th.Tensor, th.Tensor]:_
stable_baselines3/common/distributions.py, all log_prob_from_params function definitions:
        actions = self.actions_from_params(mean_actions, log_std) ->
        _actions = self.actions_from_params(mean_actions, log_std, **obs**)_
stable_baselines3/common/distributions.py, all actions_from_params function definitions:
        def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor: ->
        _def actions_from_params(self, action_logits: th.Tensor, **obs**, deterministic: bool = False) -> th.Tensor:_
stable_baselines3/common/distributions.py, all actions_from_params function definitions:
        return self.get_actions(deterministic=deterministic)
        _return self.get_actions(**obs**, deterministic=deterministic)_
stable_baselines3/sac/policies.py, line 175: 
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs) ->
        _return self.action_dist.log_prob_from_params(mean_actions, log_std, **obs**, **kwargs)_
stable_baselines3/sac/policies.py, line 170:
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs) ->
        _return self.action_dist.actions_from_params(mean_actions, log_std, **obs**, deterministic=deterministic, **kwargs)_


