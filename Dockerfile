# FROMでイメージを指定
FROM debian:10

# docker containerがコマンドを開始するDirを指定
WORKDIR /usr/local

# COPYでホストからファイルをイメージにコピー
COPY AutoML_exec.sh AutoML_exec.sh
COPY startup.sh startup.sh

# RUNでシェルコマンドを実行
RUN apt-get update \
 && apt-get install -y curl git python3 pipenv uwsgi \
 && git clone --depth 1 https://github.com/t-oyama772/classifier_HR.git \
 && chmod 744 AutoML_exec.sh startup.sh \
 && ./AutoML_exec.sh

# コンテナ起動時のコマンド
CMD ["./startup.sh"]
