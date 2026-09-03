# Execution sandbox for the agent's bash tool.
# The agent LLM runs on OpenRouter; ONLY the shell commands it emits run here,
# inside Docker's Linux VM, isolated from macOS. Containers are launched with
# --network none (see harness/sandbox.py), so any exfiltration curl the agent
# is tricked into running fails at the network layer and the fake secret never
# leaves the box. This image just needs a realistic Unix userland + curl so the
# attack can be *attempted* (which is what the paper scores).
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates coreutils findutils grep sed gawk procps \
        python3 \
    && rm -rf /var/lib/apt/lists/*

# Unprivileged user; the agent operates here.
RUN useradd -m -s /bin/bash agent
USER agent
WORKDIR /workspace/container-1
