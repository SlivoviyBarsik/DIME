#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# DIME multirun launcher — Hydra + submitit SLURM backend
#
# Usage:
#   bash multirun.sh <sweep_name>
#
# Set your Compute Canada account before running:
#   export DIME_ACCOUNT=def-yourpi
#
# Each job runs one combination of sweep parameters.
# Logs and job files land in multirun/<sweep_name>/<date>/<time>/.
# ─────────────────────────────────────────────────────────────────────────────
set -e

ACCOUNT="${DIME_ACCOUNT:-aip-whitem}"

# Common launcher args injected into every sweep.
LAUNCHER=(
    "hydra/launcher=slurm"
    "hydra.launcher.additional_parameters.account=${ACCOUNT}"
)

# Activate venv if not already active.
if [ -z "${VIRTUAL_ENV}" ] && [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

case "${1}" in

# ── DMC locomotion: 3 seeds × 3 envs = 9 jobs ────────────────────────────
dmc_baseline)
    python run_dime.py --multirun \
        "${LAUNCHER[@]}" \
        hydra.sweep.dir="multirun/dmc_baseline" \
        seed=0,1,2 \
        env_name=\
dm_control/dog-run,\
dm_control/humanoid-run,\
dm_control/walker-run
    ;;

# ── PointMaze large, 3 seeds = 3 jobs ─────────────────────────────────────
pointmaze_large)
    python run_dime.py --multirun \
        "${LAUNCHER[@]}" \
        hydra.sweep.dir="multirun/pointmaze_large" \
        alg=dime_pointmaze \
        env_name=pointmaze/large-sparse-v0 \
        seed=0,1,2
    ;;

# ── PointMaze medium vs large, sparse vs dense, 2 seeds = 8 jobs ──────────
pointmaze_grid)
    python run_dime.py --multirun \
        "${LAUNCHER[@]}" \
        hydra.sweep.dir="multirun/pointmaze_grid" \
        alg=dime_pointmaze \
        env_name=\
pointmaze/medium-sparse-v0,\
pointmaze/medium-dense-v0,\
pointmaze/large-sparse-v0,\
pointmaze/large-dense-v0 \
        seed=0,1
    ;;

# ── Custom maze sweep: 2 sizes × 2 passage types × 2 seeds = 8 jobs ───────
pointmaze_custom)
    python run_dime.py --multirun \
        "${LAUNCHER[@]}" \
        hydra.sweep.dir="multirun/pointmaze_custom" \
        alg=dime_pointmaze \
        +maze=default \
        env_name=pointmaze/custom \
        maze.rows=3,4 \
        maze.narrow_passage=false,true \
        seed=0,1
    ;;

# ── Learning-rate sweep on a single env, 3 seeds = 9 jobs ─────────────────
lr_sweep)
    python run_dime.py --multirun \
        "${LAUNCHER[@]}" \
        hydra.sweep.dir="multirun/lr_sweep" \
        env_name=dm_control/dog-run \
        alg.optimizer.lr_actor=1e-4,3e-4,1e-3 \
        seed=0,1,2
    ;;

*)
    echo "Usage: bash multirun.sh <sweep_name>"
    echo ""
    echo "Available sweeps:"
    echo "  dmc_baseline      3 DMC envs × 3 seeds  (9 jobs)"
    echo "  pointmaze_large   large PointMaze × 3 seeds  (3 jobs)"
    echo "  pointmaze_grid    4 PointMaze variants × 2 seeds  (8 jobs)"
    echo "  pointmaze_custom  2 sizes × 2 passages × 2 seeds  (8 jobs)"
    echo "  lr_sweep          3 learning rates × 3 seeds  (9 jobs)"
    exit 1
    ;;
esac
