FROM dynverse/dynwrap:py3.6

RUN pip install tensorflow

RUN pip install git+https://github.com/GPflow/GPflow

RUN pip install git+https://github.com/ManchesterBioinference/GrandPrix

LABEL version 0.1.5

ADD . /code

ENTRYPOINT python /code/run.py
