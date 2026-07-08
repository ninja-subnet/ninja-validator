# Sandbox image: the locked-down environment one miner agent runs in.
# Human-readable mirror of src/tau/sandbox/image.py, which builds the runtime image.
# The agent's own bundle is mounted at run time, and it reaches the LLM only through
# the validator proxy (no internet on the sandbox's network).
FROM python:3.11-slim

COPY src/tau/sandbox/sortdir.c /opt/tau/sortdir.c

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash git ca-certificates gcc libc6-dev \
    && gcc -O2 -shared -fPIC -o /opt/tau/libsortdir.so /opt/tau/sortdir.c -ldl -lpthread \
    && apt-get purge -y --auto-remove gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Pre-install a common agent runtime so typical agents import without network
# access (the sandbox has none). Agents needing more must vendor it in their code.
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir mini-swe-agent httpx openai

WORKDIR /work

# The container is started with `sleep <ttl>`; the harness is exec'd into it.
CMD ["bash"]
