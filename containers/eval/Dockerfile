FROM python:3.13.12-slim-bookworm@sha256:a58daefb915e1e03ad48f3ca4df8832065412c5c35cacb9d39f4229184de12b6

LABEL org.opencontainers.image.title="AuditCoder evaluation runtime"
LABEL org.opencontainers.image.description="Minimal non-root Python runtime for generated-code evaluation"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --gid 65532 auditcoder \
    && useradd --uid 65532 --gid 65532 --no-create-home \
        --shell /usr/sbin/nologin auditcoder

WORKDIR /work
USER 65532:65532

ENTRYPOINT ["python", "-I", "-B"]
CMD ["-c", "print('auditcoder evaluation image ready')"]
