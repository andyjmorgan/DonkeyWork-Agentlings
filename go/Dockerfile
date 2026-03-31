FROM golang:1.25-alpine AS build
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o /agentling ./cmd/agentling

FROM alpine:3.21
RUN apk add --no-cache bash
COPY --from=build /agentling /agentling
VOLUME /data
ENV AGENT_DATA_DIR=/data
EXPOSE 8420
ENTRYPOINT ["/agentling"]
