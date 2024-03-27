#!/bin/bash

./replica_mono.sh
./replica_rgbd.sh

./tum_mono.sh
./tum_rgbd.sh

./euroc_stereo.sh
