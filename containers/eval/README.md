# Container evaluation backend

Build the pinned, dependency-free Python evaluation image:

```bash
docker build --tag auditcoder-eval:aaai27 containers/eval
```

Select the backend before running AuditCoder:

```bash
export AUDITCODER_EXECUTION_BACKEND=container
export AUDITCODER_CONTAINER_RUNTIME=docker
export AUDITCODER_CONTAINER_IMAGE=auditcoder-eval:aaai27
```

The executor invokes Docker or Podman without a shell and enforces no network,
a read-only root filesystem, a non-root UID, dropped capabilities,
`no-new-privileges`, CPU/memory/PID limits, a bounded `/tmp`, and a read-only
mount containing only the generated program. A file-I/O execution contract
adds only its explicit `app_dir` as a writable bind mount.

The container receives no host API-key environment variables. The container
runtime client itself still runs on the host and requires access to its normal
socket. Keep that socket outside generated code and do not mount it into the
evaluation container.
