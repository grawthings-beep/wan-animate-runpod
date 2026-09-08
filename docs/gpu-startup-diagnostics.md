# GPU startup checks

## Contract

GPU names and the legacy `WAN_GPU_FAMILY` image label never veto CUDA. A driver
version string or failed NVML command alone cannot veto successful CUDA either.
The runtime preserves `CUDA_VISIBLE_DEVICES`, including UUIDs, unset and empty
values. The removed `CUDA_NORMALIZE_VISIBLE_DEVICES` setting is ignored.

Before model downloads, the pinned torch/vision/audio versions are imported and
checked. A separate process allocates a CUDA tensor, multiplies it, computes a
small FP16 matrix product, synchronizes and checks the results on CPU. This is
a startup check, not an end-to-end video generation or performance benchmark.

If CUDA fails, a separate process (15-second timeout) records device node
type/major/minor and open results, the reported kernel driver, registered NVIDIA
devices, mapped libcuda paths and `cuInit`. It never installs drivers, changes
visibility, creates device nodes, reloads modules or resets a GPU. Partial JSON
is retained if the driver call hangs. EIO and permission errors are reported
without repeatedly waiting for an unchanged failure; missing devices and unknown
CUDA failures can retry within `CUDA_READY_TIMEOUT`. Individual probe timeouts
can extend the wall-clock time beyond that retry budget.

Failures are classified as runtime-stack, device-selection, device-access,
driver-library, driver-compatibility, cuda-memory or cuda-operation. Classification
identifies the failed layer, not who caused the fault. A UVM EIO does not prove a
physically defective GPU. An image update cannot guarantee recovery of a host's
UVM driver state.

## Report and mobile UI

`GPU_DIAGNOSTICS_FILE` defaults to `/workspace/config/gpu-diagnostics.json`.
Each atomic report contains the current boot ID, image revision, allowlisted
environment, probe evidence, bounded recent attempts, category and advice.
Environment dumps, authentication URLs and model/download URLs are excluded.
Known secret environment values and HTTP(S) URLs in probe output are redacted.
The JSON includes Pod/GPU identifiers and should be shared privately.

The startup server on 8188 provides `/diagnostics.json` as an attachment named
`wan-gpu-diagnostics.json`. The page displays a Japanese download button, hides
the waiting animation on failure and says generation has stopped. Reports from
other boot IDs, malformed or oversized files are never offered. Status is reset
each boot so a prior failure or model count cannot leak into the current UI.
The shell error trap preserves an already classified GPU failure.

The download route exists only while the startup server owns port 8188. The JSON
file remains after ComfyUI takes over. A Pod's termination can destroy it; save
diagnostics and output before choosing any destructive lifecycle operation.
The failure page is held for `BOOT_FAILURE_HOLD_SECONDS` (default 900). This does
not automatically stop a billable Pod or guarantee its next host assignment.

## Images and validation

Use `loop-cu128-sha-<commit>` as the first shared configuration for 4090/5090;
`loop-cu130-sha-<commit>` remains available for compatible CUDA13 hosts. Both use
the existing pinned RunPod bases and torch2.10 stacks. Old family-named tags are
aliases to the corresponding image. No model, LoRA, video quality or download
profile has changed as part of this startup fix.

CI tests the cross-family admission logic with synthetic GPU responses, CUDA
visibility preservation, failure classification, secret removal, timed-out
driver probes, atomic report writes, stale report rejection and real HTTP JSON
downloads. Existing container CPU/node/quantization/upscale tests remain.
**GPU hardware validation is NOT RUN by this CPU-only CI.** A stack-only check
is labelled as such in JSON and cannot be presented as a successful GPU check.

## Investigation references

- [RunPod official CUDA13/4090 test configuration](https://github.com/runpod-workers/comfyui-base/blob/33dfcd67fd5dfa666bc339aae4b92cf25d292a8b/tests/comfyui/images.example.yaml)
- [PyTorch CUDA12.8 Blackwell support](https://pytorch.org/blog/pytorch-2-7/)
- [NVIDIA supported visibility/UUID formats](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#env-vars)
- [NVIDIA 580.65.06 UVM open handler](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.65.06/kernel-open/nvidia-uvm/uvm.c)
- [NVIDIA UVM status to errno mapping](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.65.06/kernel-open/nvidia-uvm/uvm_common.c)
