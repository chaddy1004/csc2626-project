#!/bin/bash

PROJ_DIR=$(pwd)
DATA_VOLUMES="
--volume ~/datasets/coco:/probdet/datasets/coco
"

echo $PROJ_DIR

CMD="docker run -it \
    --gpus all \
    --net=host \
    --privileged=true \
    --ipc=host \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="$XAUTHORITY:/root/.Xauthority:rw" \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --env LANG=C.UTF-8 \
    --hostname="inside-DOCKER" \
    --name="csc2626_project" \
    --volume $PROJ_DIR:/csc2626-project \
    --rm \
    csc2626_project bash
"

echo $CMD
eval $CMD
