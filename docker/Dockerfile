FROM korekontrol/ubuntu-java-python3

LABEL maintainer="theunsignedfairon@gmail.com"


RUN apt-get update \
  && apt-get install --no-install-recommends -y \
    bash \
    build-essential \
    curl \
    gettext \
    git \
    libpq-dev \
    wget \
    nano \
    $BUILD_ONLY_PACKAGES \
  && apt-get remove -y $BUILD_ONLY_PACKAGES \
  && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
  && apt-get clean -y && rm -rf /var/lib/apt/lists/* \
  && pip install --upgrade pip

WORKDIR /app

COPY ./docker/requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

COPY ./code2vec ./code2vec/
COPY ./web_service/model.py ./web_service/state_dict.uu ./docker/predict.py ./src/
