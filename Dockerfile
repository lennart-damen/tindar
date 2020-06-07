# Use existing image as base
FROM python:3.7.6-slim-buster

RUN mkdir /app
WORKDIR /app/

# GCC build issue: https://github.com/docker-library/python/issues/318
RUN apt-get update \
    && apt-get -y install gcc

# Download and install dependencies
COPY . .
RUN pip install -r requirements.txt

ENV DOCKER=1
ENV PORT=8080
ENV HOST=0.0.0.0
ENV FLASK_ENV=production

# Run flask app
EXPOSE 8080
ENTRYPOINT ["python"]
CMD ["run.py"]