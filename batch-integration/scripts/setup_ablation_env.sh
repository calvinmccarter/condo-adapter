#!/usr/bin/env bash
# Bootstrap a machine to run the condo agglomerative-ranking ablation sweep.
# Idempotent: skips anything already present. Assumes `condo-adapter` and
# `task_batch_integration` are cloned side by side under $ROOT.
#
#   ROOT/
#   ├── condo-adapter/            (this repo)
#   ├── task_batch_integration/   (fork with src/methods/condo/condo_runner.py)
#   ├── .venv/                    (created here: main GPU env, editable condo)
#   ├── .venv-kbet/               (created here: numpy<2 + rpy2 + anndata2ri 1.3.1)
#   └── fullbench/datasets/<ds>/{dataset,solution}.h5ad   (downloaded here)
#
# Usage:  bash condo-adapter/batch-integration/scripts/setup_ablation_env.sh [ROOT]
# Needs:  sudo (apt for R), a CUDA GPU, ~25 GB disk for data, public S3 access.
set -uo pipefail

ROOT="${1:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$ROOT"
echo ">> ROOT=$ROOT"
[ -d condo-adapter ] || { echo "ERR: $ROOT/condo-adapter not found"; exit 1; }
[ -d task_batch_integration ] || { echo "ERR: $ROOT/task_batch_integration not found"; exit 1; }

# --- uv -----------------------------------------------------------------
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# --- main GPU venv ------------------------------------------------------
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.11 .venv
  uv pip install --python .venv/bin/python \
    torch "anndata>=0.10" scanpy h5py scipy numpy pandas scikit-learn \
    "miceforest<6.0.0" pytorch-minimize tqdm scib -e ./condo-adapter
fi
.venv/bin/python -c "import torch;assert torch.cuda.is_available()" \
  && echo ">> .venv OK (CUDA available)" || echo "!! WARNING: CUDA not available in .venv"

# --- R + kBET (apt + user R library) ------------------------------------
if ! command -v Rscript >/dev/null 2>&1; then
  sudo -n apt-get update -qq
  sudo -n DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    r-base r-base-dev libtirpc-dev libcurl4-openssl-dev libssl-dev libxml2-dev
fi
export R_LIBS_USER="$HOME/R/x86_64-pc-linux-gnu-library/4.3"
mkdir -p "$R_LIBS_USER"
Rscript -e '.libPaths(Sys.getenv("R_LIBS_USER"));
  if (!requireNamespace("kBET", quietly=TRUE)) {
    options(repos=c(CRAN="https://cloud.r-project.org"), Ncpus=8);
    install.packages(c("remotes","FNN"));
    remotes::install_github("theislab/kBET", upgrade="never",
                            dependencies=c("Depends","Imports","LinkingTo")) };
  cat("kBET installed:", requireNamespace("kBET", quietly=TRUE), "\n")'

# --- kbet venv (numpy<2 + rpy2 + anndata2ri 1.3.1; scib 1.1.7 needs .activate) ---
if [ ! -x .venv-kbet/bin/python ]; then
  uv venv --python 3.11 .venv-kbet
  uv pip install --python .venv-kbet/bin/python \
    "numpy<2" "scipy<=1.13" pandas "anndata>=0.10" scanpy scib h5py \
    "rpy2>=3.5,<3.6" "anndata2ri==1.3.1"
fi

# --- datasets (log_cp10k = condo's preferred_normalization) -------------
BASE=s3://openproblems-data/resources/task_batch_integration/datasets/cellxgene_census
mkdir -p fullbench/datasets
for ds in dkd gtex_v9 hypomap immune_cell_atlas mouse_pancreas_atlas tabula_sapiens; do
  for f in dataset solution; do
    dst="fullbench/datasets/$ds/$f.h5ad"
    [ -s "$dst" ] && continue
    mkdir -p "fullbench/datasets/$ds"
    echo ">> downloading $ds/$f.h5ad"
    aws s3 cp --no-sign-request --only-show-errors "$BASE/$ds/log_cp10k/$f.h5ad" "$dst"
  done
done

cat <<EOF

>> SETUP COMPLETE. Before running the sweep, export:
   export CONDO_KBET_PYTHON=$ROOT/.venv-kbet/bin/python
   export R_LIBS_USER=$R_LIBS_USER
   export R_HOME=/usr/lib/R

>> Then, e.g. (big-memory GPU box, immune + tabula, WITH kbet):
   $ROOT/.venv/bin/python \\
     condo-adapter/batch-integration/scripts/run_ablation_sweep.py \\
     --datasets immune_cell_atlas tabula_sapiens
EOF
