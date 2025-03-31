# NL Search

## Run locally

```bash
docker run -v /Users/katso/athena/text-to-sql/trained-models:/trained-models -e GUNICORN_CMD_ARGS="--workers=1" -p 80:80 api_nl_search
```
