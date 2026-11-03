"""Code the environments may share: `sandbox` (one container per episode
with a host-visible workspace), `container` (the podman or Apptainer
runtime under it) and `sources` (where problems come from and how answers
are scored). None of it is required by the contract in `envs`, and it
is not an environment itself.
"""
