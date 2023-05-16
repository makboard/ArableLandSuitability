#!/bin/bash
docker run -it -v /mnt/data/lukashevich/igran:/workspace --gpus "all" -w=/workspace wind_dev
