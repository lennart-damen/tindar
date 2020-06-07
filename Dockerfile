# Use existing image as base
FROM python:3.7.6-slim-buster
# FROM python:3.7-alpine

# GCC build issue: https://github.com/docker-library/python/issues/318
RUN apt-get update \
    && apt-get -y install gcc

# RUN apk update \
    # && apk add build-base

# Use app directory to put tindar app
RUN mkdir /app
WORKDIR /app/

# Download and install dependencies
COPY . .
RUN pip install -r requirements.txt

ENV DOCKER=1
ENV PORT=8080
ENV HOST=0.0.0.0
ENV FLASK_ENV=production

# Run flask app
EXPOSE 8080
CMD ["gunicorn"  , "-b", "0.0.0.0:8080", "flask_application:app"]