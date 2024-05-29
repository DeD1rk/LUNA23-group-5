FROM pytorch/pytorch

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

COPY --chown=algorithm:algorithm requirements-container.txt /opt/algorithm/requirements.txt

USER algorithm
WORKDIR /opt/algorithm

RUN python -m pip install --user -U pip && \ 
    python -m pip install --user --no-cache-dir -r requirements.txt

ADD --chown=algorithm:algorithm luna /opt/algorithm/luna
COPY --chown=algorithm:algorithm checkpoints /opt/algorithm/checkpoints

ENV PATH="/home/algorithm/.local/bin:${PATH}"
ENTRYPOINT python -m luna.process $0 $@
