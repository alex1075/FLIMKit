FROM debian:latest
LABEL maintainer="Alexander Hunt <alexander.hunt@ed.ac.uk>"
LABEL description="Docker image setup to use the FLIM fitting script for Leica SP8 microscope data analysis in the cloud. The image includes Python 3 and necessary dependencies for running the fitting script."
ENV TZ=Europe/London \
    DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip
WORKDIR /app
COPY . /app
RUN pip3 install -r requirements.txt
CMD ["./main.py"]