FROM tensorflow/tensorflow:1.13.1-gpu-py3

WORKDIR /app

ENV BATCH_SIZE 10000
ENV EPOCHS 500000

COPY . /app
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update -y && \
    apt-get install -y gcc-4.9 && \
    apt-get install --only-upgrade -y libstdc++6

RUN pip install --upgrade pip

RUN pip install --requirement /app/requirements.txt && \
    curl -L https://julialang-s3.julialang.org/bin/linux/x64/1.1/julia-1.1.1-linux-x86_64.tar.gz -o julia.tar.gz && \
    tar xzf julia.tar.gz && \
    mv julia-1.1.1/ /julia 

ENV PATH="/julia/bin:${PATH}"

RUN julia -e 'using Pkg; Pkg.add("CSV")'
RUN julia -e 'using Pkg; Pkg.add("DataFrames")'
RUN julia -e 'using Pkg; Pkg.add("DiffResults")'
RUN julia -e 'using Pkg; Pkg.add("DifferentialEquations")'
RUN julia -e 'using Pkg; Pkg.add("Distributions")'
RUN julia -e 'using Pkg; Pkg.add("ForwardDiff")'
RUN julia -e 'using Pkg; Pkg.add("LinearAlgebra")'
RUN julia -e 'using Pkg; Pkg.add("MeshCat")'
RUN julia -e 'using Pkg; Pkg.add("MeshCatMechanisms")'
RUN julia -e 'using Pkg; Pkg.add("Polynomials")'
RUN julia -e 'using Pkg; Pkg.add("PyCall")'
RUN julia -e 'using Pkg; Pkg.add("Random")'
RUN julia -e 'using Pkg; Pkg.add("RigidBodyDynamics")'

RUN julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'
WORKDIR /app

ENTRYPOINT [ "python", "main.py" ]
