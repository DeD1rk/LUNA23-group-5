FROM pytorch/pytorch



RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/

RUN python -m pip install --user -r requirements.txt

ADD --chown=algorithm:algorithm luna /opt/algorithm/luna
COPY --chown=algorithm:algorithm checkpoints /opt/algorithm/checkpoints

ENTRYPOINT python -m luna.process $0 $@
