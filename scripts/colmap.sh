#!/bin/bash

../bin/train_colmap \
    ../cfg/colmap/gaussian_splatting.yaml \
    /home/rapidlab/programs/NeuralSLAM_ws/gaussian-splatting-materials/tandt_db/db/drjohnson \
    ../results/colmap/drjohnson \
    no_viewer