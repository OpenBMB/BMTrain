---
name: Bug report
about: Create a report to help us improve
title: "[BUILD ERR]"
labels: Build ERR
assignees: ''

---

### Build Error Report

**Description of the Build Error**
Please provide a clear and concise description of what the build error is.

**To Reproduce**
Steps to reproduce the behavior:
1. Describe the steps you took to trigger the build error.
2. Make sure to include any commands you executed or files you modified.

**Expected Behavior**
Provide a clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment Information**
- Operating System version:
- GCC version:
- CUDA version:
- CMake version:
- Pip version:

**Installation Method**
Please indicate if the error occurred during source code installation or when using the `pip install .whl` method:

**If Source Code Installation:**
- Confirm that you used `pip install .` for installation and not `python setup.py`.

**If Pip Install .whl Method:**
- Provide the versions of pip, CUDA, GCC, and the operating system.
- Confirm that you have installed `nvidia-nccl-cu11>=2.14.3` from PyPI.

**Full Error Traceback**
Provide the complete error traceback.

**Additional Information**
Provide any other relevant context or information about the problem here.

**Confirmation**
Please confirm that you have reviewed all of the above requirements and verified the information provided before submitting this report.
