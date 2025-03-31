FROM darelab.docker.imsi.athenarc.gr/darelab/base-py11-cpu:latest

ARG wheel=api_nl_search-0.1.0-py3-none-any.whl

WORKDIR /code

# Install dependencies early to keep this step cached
COPY ./dist/$wheel /code/$wheel
RUN pip install --no-cache-dir --upgrade --no-deps /code/$wheel

COPY ./fc4eosc_schema.json /code/
COPY ./static /code/static
COPY ./templates /code/templates
COPY ./third_party /code/third_party
COPY .env /code/.env

RUN unzip $wheel 'darelabdb/*'

EXPOSE 80
CMD ["uvicorn", "darelabdb.api_nl_search.app.llama_main:app", "--host", "0.0.0.0", "--port", "80"]
