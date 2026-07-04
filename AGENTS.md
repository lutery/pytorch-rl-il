# AGENTS.md — PyTorch-RL-IL

## Architecture

This is a PyTorch library for RL/IL research, forked from Autonomous Learning Library (ALL).
It adds distributed sampling via Ray and wraps the [cpprb](https://ymd_h.gitlab.io/cpprb/) replay buffer.

### Top-level layout

| Dir | Purpose |
|---|---|
| `rlil/agents/` | Algorithm implementations (SAC, PPO, DDPG, TD3, GAIL, etc.) |
| `rlil/presets/continuous/` | Factory functions wiring agents ↔ models ↔ buffers (entry point for training) |
| `rlil/approximation/` | Wrappers around `nn.Module`: QContinuous, VNetwork, FeatureNetwork, etc. |
| `rlil/environments/` | `GymEnvironment` wrapper; custom PyBullet envs for domain-shift IL research |
| `rlil/experiments/` | `Experiment` + `Trainer` — the main run loop |
| `rlil/samplers/` | `AsyncSampler` using `ray.remote` workers for parallel experience collection |
| `rlil/memory/` | `ExperienceReplayBuffer` (cpprb wrapper) + wrappers for GAE, GAIL, AIRL, SQIL |
| `rlil/nn/` | Custom `nn.Module` subclasses: `RLNetwork`, `NoisyLinear`, `Dueling`, `Flatten`, etc. |
| `rlil/initializer.py` | **Global mutable state** for device, seed, writer, logger, replay buffer, on/off-policy mode, n-step, Ape-X flags |
| `rlil/utils/` | `ExperimentWriter` (TensorBoard) and plotting helpers |
| `scripts/` | Run scripts: `online.py`, `offline.py`, `online_il.py`, `record_trajectory.py`, `plot.py`, `watch_continuous.py` |
| `tests/` | pytest tests mirroring `rlil/` structure; benchmarks in `tests/benchmark/` |

### Critical: `rlil.initializer` global state

This module uses module-level globals for cross-cutting concerns. Presets call setters during construction. This is the most non-obvious architectural pattern:

- **`set_device()` / `get_device()`** — presets call `get_device()` to place models
- **`set_seed()` / `call_seed()`** — called by `Experiment.__init__` and `AsyncSampler.__init__`
- **`set_replay_buffer()` / `get_replay_buffer()`** — presets set the buffer; agents and samplers retrieve it globally
- **`enable_on_policy_mode()` / `is_on_policy_mode()`** — PPO enables it; the Trainer uses it to decide whether to train inside the sample loop
- **`set_n_step()` / `get_n_step()`** — presets set it; `LazyAgent.__init__` reads it to configure its per-worker replay buffer
- **`enable_apex()` / `use_apex()`** — presets enable it; used by AsyncSampler for priority-based sampling

When writing or debugging presets, always check which global flags they set. If you see unexpected behavior, suspect stale global state.

### Agent ↔ LazyAgent pattern

Every agent implements two interfaces:
1. **`Agent`** (lives in the main process) — `act()`, `train()`, `make_lazy_agent()`
2. **`LazyAgent`** (shipped to Ray workers) — `act()`, `set_replay_buffer()`, `compute_priorities()`

`Trainer` calls `agent.make_lazy_agent()` and passes the result to `AsyncSampler.start_sampling()`. The LazyAgent collects transitions into its own per-worker replay buffer, then `AsyncSampler.store_samples()` retrieves and merges them into the global buffer.

## Developer commands

```bash
# Install (torch must be pre-installed; not in setup.py core deps)
pip install -e .

# Run all tests (skip benchmarks)
make test                          # or: pytest -v --benchmark-skip

# Run benchmarks only
make benchmark                     # or: pytest -v --benchmark-only

# Run a single test
pytest tests/presets/online_continuous_test.py::test_sac -v --benchmark-skip

# Format code
make autopep8                      # or: autopep8 --in-place --recursive .

# TensorBoard
make tensorboard                   # or: tensorboard --logdir runs
```

## Training invocation

All training launches through scripts. The core pattern:

```bash
# Online RL
python scripts/continuous/online.py <env> <agent> --train_minutes 60 --num_workers 5 --exp_info "description"

# Offline IL / RL (requires a demo directory with transitions.pkl)
python scripts/continuous/offline.py <env> <agent> <path_to_demo_dir>

# Online IL
python scripts/continuous/online_il.py <env> <il_agent> <base_agent> <path_to_demo_dir>
```

**Env names**: Use short names like `ant`, `hopper`, `walker`, `humanoid`, `cheetah`, `lander`, `pendulum`, `mountaincar` (see `ENVS` dict in `rlil/environments/__init__.py`). You can also pass the raw Gym ID directly.

**Agent names**: Match preset function names: `ppo`, `sac`, `td3`, `ddpg`, `vac`, `noisy_td3`, `bc`, `vae_bc`, `bcq`, `bear`, `brac`, `gail`, `airl`, `sqil`, `rs_mpc`.

## Presets are the integration point

Presets are functions that return a `_fn(env)` factory. They:
1. Set global state flags (`enable_on_policy_mode`, `set_n_step`, `set_replay_buffer`, etc.)
2. Build models from `rlil/presets/continuous/models.py`
3. Wrap models in approximation objects (`VNetwork`, `QContinuous`, etc.)
4. Build the replay buffer and optional wrappers (`GaeWrapper`, `GailWrapper`, etc.)
5. Construct and return the Agent

To add a new algorithm: create a preset in `rlil/presets/continuous/`, an agent in `rlil/agents/`, and update both `__init__.py` files.

## Testing conventions

- **`conftest.py`** auto-applies `set_seed(0)`, `enable_debug_mode()`, and `reset_action_space` to every test
- `use_cpu` fixture forces CPU device; some tests (SAC, Ape-X) explicitly use it for GPU memory reasons
- `env_validation(agent_fn, env)` — runs 2 episodes end-to-end through an agent
- `trainer_validation(agent_fn, env)` — exercises the full Agent/LazyAgent/Sampler pipeline
- Benchmark tests in `tests/benchmark/` are skipped by default; use `--benchmark-only` to run them
- Trainer tests must call `ray.init(include_webui=False, ignore_reinit_error=True)` — the `trainer_test.py` fixture handles this

## Key quirks and gotchas

- **`ray.init()` is required** before running any training script or test that uses `AsyncSampler`. Set `num_workers=0` to skip the sampler entirely.
- **`append_time=True`** must be passed to `GymEnvironment` when the env is a `TimeLimit` wrapper (most Gym envs are). Without this, the state doesn't include the timestep feature.
- **PyBullet imports are side-effectful**: `import pybullet; import pybullet_envs` is needed at the top of training scripts to register Bullet envs with Gym. These imports are present in `scripts/continuous/online.py`.
- **Offline RL is experimental**: README states BCQ/BEAR/BRAC implementations "should be incorrect" and results don't match original papers. Treat them as work-in-progress.
- **Ape-X is unstable**: Per [issue #4](https://github.com/syuntoku14/pytorch-rl-il/issues/4), the Ape-X implementation is sensitive to mini-batch size and uses episodic (not continuous) training.
- **Torch is not a core dependency**: `setup.py` lists it in `extras_require["pytorch"]`, not `install_requires`. The Makefile installs it via conda separately.
- **`Action.action_space()` is a class-level global** set by `Action.set_action_space()` during `GymEnvironment.__init__`. This means only one action space can be active at a time — tests reset it via `reset_action_space` fixture.
- **Demos use `tsp`** (task-spooler): batch scripts in `scripts/` use `tsp` to queue jobs. This is not required — it's a convenience for batch experiments.
- **No CI configured** — there are no GitHub Actions workflows or similar.
- **`runs/` is the output directory** (gitignored). Results are organized as `runs/<exp_info>/<env>/<agent_with_id>/`.