PhaseKV 4090D: first-cooling CPU measurement, no CUPTI

Upload PhaseKV_4090D_cooling_cpu.zip into /data, then run:
  cd /data
  unzip -o PhaseKV_4090D_cooling_cpu.zip
  bash /data/phasekv_4090d_cooling_cpu/start.sh

Keep this terminal open. The script prints a new result directory under
/data/phasekv-results/4090d_cooling_cpu_XXXXXXXX. Wait for FINISHED, then
download cpu_summary.json and run.log from that directory.

Uses the existing /data/phasekv_4090d_speed frozen package and
/data/phasekv-native-venv/bin/python. Checks the original benchmark GPU UUID,
versions, code, input and model metadata. Loads the full 32-layer model once.
All runs use the same 8192-token input, 256 greedy output tokens, GQA on,
70% prompt retention and 80% generated-KV retention. Every output hash must
match the previous 4090D benchmark. Production cache/kernel files are unchanged.

One warmup per arm is excluded. The default is four paired repetitions,
alternating plain -> host_timed and host_timed -> plain. Each arm starts a
fresh generation and cache. --pairs 2 or --pairs 6 is also supported.

Only three non-overlapping functions are timed during decode steps 129-159:
  cache_append: append_hot, 32 calls per decode step
  cache_finish: finish_decode_step, 1 call per decode step (all 32 layers)
  workspace: ensure_workspace, 32 calls per decode step
Each wrapper reads perf_counter_ns and thread_time_ns before/after the call.
Counters are preallocated. No profiler, CUDA events, activity callbacks or
GPU synchronization inside the measured functions. Synchronization occurs
at the same window boundaries in both arms, as in the previous measurement.

Windows:
  reference129: decode 129, hot length 129, no generated cold block
  before: decode 130-143, no generated cold block
  cooling144: decode 144, first cooling AFTER attention
  first_cold145: decode 145, first use of the cold block/descriptors
  after: decode 146-159, same hot lengths as 130-143, one cold block
Step 1 consumes the first generated output token; it is not output token 1.

Units are milliseconds summed over all model layers, per decode step.
host_elapsed_ms is host call duration, including implicit CUDA API waiting
and OS descheduling. thread_cpu_ms is CPU time of the calling thread,
excluding waiting and other threads. Their difference is NOT necessarily
GPU waiting. CPU and GPU times overlap and must not be added. The two CPU
metrics also overlap and must not be added to each other.

The report stores per-step counters, per-run windows, paired overhead and
within-generation contrasts. Differences are computed within each run/pair
before taking the median. Clock implementations and observed granularity are
recorded; the production launcher rejects coarse clocks before loading weights.
The perturbation check uses absolute paired median
wall-time differences <=3% for multi-step windows and <=5% for single steps.
This is an empirical check, not a statistical guarantee. Negative paired
differences indicate timing variability; they are not negative timer cost.
If the check exceeds its budget, results remain available but must be reviewed
before treating the stage durations as representative of normal execution.
