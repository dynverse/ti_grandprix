FROM dynverse/dynwrappy:v0.1.0

RUN pip install tensorflow

RUN pip install git+https://github.com/GPflow/GPflow

RUN pip install git+https://github.com/ManchesterBioinference/GrandPrix

COPY definition.yml example.h5 run.py /code/

ENTRYPOINT ["/code/run.py"]
