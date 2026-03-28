FROM python:3.12-slim

COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir .

VOLUME /data
ENV AGENT_DATA_DIR=/data
EXPOSE 8420

ENTRYPOINT ["agentling"]
