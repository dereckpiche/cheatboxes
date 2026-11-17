"""Code the environments may share: `sandbox` (one container per episode
with a host-visible workspace), `container` (the podman or Apptainer
runtime under it), `programs` (Python code run on calls, each program in
its own sandbox) and `forum` (a folder every episode of a run shares). None of it is required by the contract in `envs`, and it
is not an environment itself.
"""
