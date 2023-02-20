# syntax=docker/dockerfile:1.4
FROM --platform=$BUILDPLATFORM ubuntu AS builder

FROM builder as dev-envs

USER root

RUN apt-get update && \
    apt-get install -y --upgrade git python3 python3-pip gcc g++ gfortran cmake && \
    rm -rf /var/lib/apt/lists/*;

RUN useradd -ms /bin/bash apprunner

# install Docker tools (cli, buildx, compose)
COPY --from=gloursdocker/docker / /

USER apprunner

WORKDIR /app

ENV FLASK_APP=app.py

COPY requirements.txt /app

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]