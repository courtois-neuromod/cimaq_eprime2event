#!/bin/bash

INPUT_FOLDER="/home/labopb/Documents/Marie/neuromod/CIMAQ/event_files/V10/taskfiles"
OUTPUT_FOLDER="/home/labopb/Documents/Marie/neuromod/CIMAQ/cimaq_eprime2event/test"

# launch job
python -m qc_eventfiles \
        --idir "${INPUT_FOLDER}" \
        --odir "${OUTPUT_FOLDER}"
