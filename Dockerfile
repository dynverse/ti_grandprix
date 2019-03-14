FROM dynverse/dynwrappy:v0.1.0

ARG GITHUB_PAT

RUN pip install tensorflow

RUN pip install git+https://github.com/GPflow/GPflow

RUN pip install git+https://github.com/ManchesterBioinference/GrandPrix

COPY definition.yml run.py example.sh /code/

ENTRYPOINT ["/code/run.py"]
