import torch

from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler


def test_sde_scheduler_zero_sigma_step_is_finite():
    scheduler = FlowMatchSDEDiscreteScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(2, device="cpu")
    scheduler.sigmas = torch.tensor([1.0, 0.0, 0.0])
    scheduler.timesteps = torch.tensor([1000.0, 0.0])
    scheduler._step_index = 1

    sample = torch.randn(2, 3, 4, 4)
    model_output = torch.randn_like(sample)

    prev_sample, log_prob, prev_sample_mean, std_dev_t = scheduler.step(
        model_output,
        scheduler.timesteps[1],
        sample,
        noise_level=0.0,
        return_logprobs=True,
        return_dict=False,
    )

    assert torch.isfinite(prev_sample).all()
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(prev_sample_mean).all()
    assert torch.isfinite(std_dev_t).all()
    torch.testing.assert_close(prev_sample, sample)
