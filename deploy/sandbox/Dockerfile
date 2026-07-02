# Sandbox image: the locked-down environment one miner agent runs in.
# Kept minimal — Python + git + a couple of common agent runtimes. The agent's own
# single-file code is copied in at run time (not baked here), and it reaches the LLM
# only through the validator proxy (no internet on the sandbox's network).
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Pre-install a common agent runtime so typical agents import without network
# access (the sandbox has none). Agents needing more must vendor it in their code.
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir mini-swe-agent httpx openai

WORKDIR /work

# The container is started with `sleep <ttl>`; the harness is exec'd into it.
CMD ["bash"]
