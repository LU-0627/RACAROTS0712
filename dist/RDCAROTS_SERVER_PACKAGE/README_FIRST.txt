RDCAROTS Server Deployment Package

QUICK START:
1. Extract this package on your Linux server
2. Read RUN_ON_OFFLINE_SERVER.md for complete instructions
3. Set environment variables (see below)
4. Run scripts/server/RUN_ALL_SERVER.sh

ENVIRONMENT VARIABLES:
export PROJECT_ROOT=$(pwd)/project
export DATA_ROOT=/path/to/your/data
export OUTPUT_ROOT=$PROJECT_ROOT/results/rd_carots
export PYTHON_BIN=python
export CUDA_VISIBLE_DEVICES=0

FIRST COMMAND:
cd project
bash scripts/server/00_check_environment.sh

For detailed instructions, see RUN_ON_OFFLINE_SERVER.md

NO INTERNET REQUIRED - This package contains everything needed for offline deployment.
