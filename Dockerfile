FROM dynverse/dynwrap:py3.6

LABEL version 0.1.2

RUN pip install tensorflow

RUN pip install git+https://github.com/GPflow/GPflow

RUN pip install git+https://github.com/ManchesterBioinference/GrandPrix

ADD . /code

ENTRYPOINT python /code/run.py
