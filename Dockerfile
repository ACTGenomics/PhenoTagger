FROM tensorflow/tensorflow:2.12.0-gpu

# 安裝基本工具 + Java（OpenJDK 11）
RUN apt-get update && apt-get install -y \
    git unzip openjdk-11-jdk \
    && rm -rf /var/lib/apt/lists/*

# 設定 JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# 複製 requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

# 下載所有常用的 NLTK 資源（包括 punkt, wordnet, pos_tag 等）
RUN python -c "import nltk; nltk.download('popular', quiet=False)"

# 複製程式碼
COPY . /PhenoTagger
WORKDIR /PhenoTagger/src/