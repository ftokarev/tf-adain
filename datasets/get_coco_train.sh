#!/bin/bash
set -e


mkdir -p datasets/coco
cd datasets/coco
wget -c http://msvocds.blob.core.windows.net/coco2014/train2014.zip
unzip train2014.zip
rm train2014.zip
