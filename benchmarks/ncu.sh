#!/usr/bin/env bash
# Nsight Compute profiling of the custom kernels.
#
#   bash benchmarks/ncu.sh          # summary CSV: bandwidth, occupancy, duration
#   bash benchmarks/ncu.sh full     # full report (open the .ncu-rep in the Nsight Compute GUI)
#
# On Colab you may need: apt-get install -y nsight-compute  (or use the ncu shipped with the CUDA toolkit)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p benchmarks/results

KERNELS='regex:layernorm|bias_gelu|softmax|flash_attention'

if [[ "${1:-}" == "full" ]]; then
  ncu --kernel-name "$KERNELS" --launch-skip 0 --launch-count 20 --set full \
      -o benchmarks/results/kernels -f python benchmarks/profile_kernels.py --iters 1
  echo "wrote benchmarks/results/kernels.ncu-rep"
else
  ncu --kernel-name "$KERNELS" --launch-count 20 --csv \
      --metrics gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum.per_second,sm__warps_active.avg.pct_of_peak_sustained_active,sm__maximum_warps_per_active_cycle_pct,launch__registers_per_thread,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
      python benchmarks/profile_kernels.py --iters 1 | tee benchmarks/results/ncu_metrics.csv
  echo
  echo "Key columns:"
  echo "  dram__throughput...pct_of_peak       -> '% of peak memory bandwidth' for the README"
  echo "  sm__warps_active...pct_of_peak       -> achieved occupancy"
  echo "  sm__maximum_warps_per_active_cycle   -> theoretical occupancy"
  echo "  l1tex__data_bank_conflicts...ld      -> shared-memory bank conflicts (should be ~0 for flash_attention)"
fi
