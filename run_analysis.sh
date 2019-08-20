
# only supported for SwDA

CURRENT_DIR=$(pwd)
VAL_INDEX_PATH="${CURRENT_DIR}/data/swda_val.txt"
DATA_PATH="${CURRENT_DIR}/data/swda_val_data.csv"
OUTPUT_PATH="${CURRENT_DIR}/data/BiLSTM-RAM_SWDA_comm_ouptut.csv"
SAVE_PATH="${CURRENT_DIR}/data/BiLSTM-RAM_SWDA_0.805_analysis.csv"

python analyze3.py \
    --output_path="${OUTPUT_PATH}" \
    --data_path="${DATA_PATH}" \
    --save_path="${SAVE_PATH}" \
