FROM tensorflow/tensorflow:1.13.1-gpu-py3

WORKDIR /app

ENV BATCH_SIZE 10000
ENV EPOCHS 500000

COPY . /app
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update -y && \
    apt-get install -y gcc-4.9 && \
    apt-get install --only-upgrade -y libstdc++6

RUN pip install --requirement /app/requirements.txt && \
    curl -L https://julialang-s3.julialang.org/bin/linux/x64/1.1/julia-1.1.1-linux-x86_64.tar.gz -o julia.tar.gz && \
    tar xzf julia.tar.gz && \
    mv julia-1.1.1/ /julia 

ENV PATH="/julia/bin:${PATH}"

ENTRYPOINT [ "python", "main.py" ]