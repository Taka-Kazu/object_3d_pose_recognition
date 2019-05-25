#!/bin/bash

IMAGE_NAME=object_3d_pose_recognition
KITTI_DIR=$1
if [[ "$1" = "" ]]
    then
        echo "kitti path is requierd!"
        echo "e.g. $HOME/dataset/kitti"
        exit 0
fi

xhost +

SCRIPT_DIR=$(cd $(dirname $0); pwd)

nvidia-docker run -it --rm \
  --privileged \
  --runtime=nvidia \
  --env=QT_X11_NO_MITSHM=1 \
  --env="DISPLAY" \
  --volume="/etc/group:/etc/group:ro" \
  --volume="/etc/passwd:/etc/passwd:ro" \
  --volume="/etc/shadow:/etc/shadow:ro" \
  --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
  --net="host" \
  --volume="$SCRIPT_DIR/:/root/$IMAGE_NAME/" \
  --volume="$KITTI_DIR/:/root/$IMAGE_NAME/dataset/kitti/" \
  -p 8008:8008 \
  $IMAGE_NAME
