FROM tensorflow/tensorflow:1.9.0-gpu-py3

COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt

COPY ./julia/ /julia/

ENV PATH="/julia/bin:${PATH}"