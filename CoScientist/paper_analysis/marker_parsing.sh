#!/bin/bash

INPUT_DIR="$1"
OUTPUT_DIR="$2"
WORKERS="$3"

DEFAULT_LANG=ru
export DEFAULT_LANG

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist"
    exit 1
fi

ulimit -n 4096

marker "$INPUT_DIR" --workers $WORKERS --output_dir "$OUTPUT_DIR" --output_format html --max_tasks_per_worker 5  --skip_existing