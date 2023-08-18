---
name: Bugs
about: Any Bugs happen in runtime.
title: "[BUG]"
labels: bug
assignees: ''
### Bug Report

**Description of the Bug**
Please provide a clear and concise description of what the bug is.

**Environment Information**
- GCC version:
- Torch version:
- Linux system version:
- CUDA version:
- Torch's CUDA version (as per `torch.cuda.version()`):

**To Reproduce**
Please provide the following details to reproduce the behavior:
1. Describe your environment setup, including any specific version requirements.
2. Clearly state the steps you took to trigger the error, including the specific code you executed.
3. Identify the file and line number where the error occurred, along with the full traceback of the error. Make sure to have `NCCL_DEBUG=INFO` and `CUDA_LAUNCH_BLOCKING=True` set to get accurate debug information.

**Expected Behavior**
Describe what you expected to happen when you executed the code.

**Screenshots**
If applicable, please add screenshots to help explain your problem.

**Additional Information**
Provide any other relevant context or information about the problem here.

**Confirmation**
Please confirm that you have reviewed all of the above requirements and verified the information provided before submitting this issue.
