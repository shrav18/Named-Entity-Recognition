FROM public.ecr.aws/bitnami/python:3.8

USER root

WORKDIR /app

ENV PYTHONPATH "${PYTHONPATH}:${WORKDIR}"

COPY inviz/requirements.txt ./
RUN pip install -r requirements.txt

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .

CMD ["python", "./src/main.py"]
