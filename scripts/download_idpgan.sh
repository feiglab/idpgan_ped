#!/bin/bash

set -e

wget https://github.com/feiglab/idpgan/archive/refs/heads/main.zip -O idpgan.zip
unzip -o idpgan.zip
rm idpgan.zip

mv idpgan-main/idpgan .
mv idpgan-main/data/*pt data/
rm -r idpgan-main