# Compression Artifacts

## Overview
This project compresses video files and handles compression artifacts. Ensure all dependencies are installed and data paths are correctly configured before running.

## Setup

### Prerequisites
Have all libraries needed.

### Configuration
Modify the data paths at the beginning of the jupyter notebook:

## Usage

1. Update the data paths in the configuration block
2. Place your input video files in right data path
3. Run the blocks one by one and you can get an entire workflow
    ```
    input videos -> compressed -> filtering -> mask extraction -> subtitle removal -> evaluation
    ```
4. Results will be saved to output paths you set

## Notes

- Check file formats are supported before processing
