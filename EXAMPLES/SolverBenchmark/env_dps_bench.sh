#!/usr/bin/env bash
# Environment for DistributedPythonSparse SolverBenchmark runs.
# Source this before launching sweeps:
#   source EXAMPLES/SolverBenchmark/env_dps_bench.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Intel oneAPI (MKL + Intel MPI matching build-mp/Release/opensees.so)
if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
fi
export LD_LIBRARY_PATH="/opt/intel/oneapi/mkl/latest/lib:/opt/intel/oneapi/mpi/latest/lib:${LD_LIBRARY_PATH:-}"
export MPI_RUN="${MPI_RUN:-/opt/intel/oneapi/mpi/latest/bin/mpirun}"

# conda env with cupy + scikit-umfpack + nvmath-python + openseespy_solvers>=0.2.0
# (editable: pip install -e /home/garaujor/openseespy-solvers)
# shellcheck disable=SC1091
source /home/garaujor/miniconda3/etc/profile.d/conda.sh
conda activate openseespy-solvers

export PYTHONPATH="${REPO_ROOT}/build-mp/Release${PYTHONPATH:+:${PYTHONPATH}}"

echo "[env_dps_bench] python=$(command -v python)"
echo "[env_dps_bench] mpirun=${MPI_RUN}"
echo "[env_dps_bench] PYTHONPATH=${PYTHONPATH}"
