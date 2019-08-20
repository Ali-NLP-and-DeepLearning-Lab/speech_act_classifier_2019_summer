CURRENT_DIR=$(pwd)
DATA_PATH="${CURRENT_DIR}/data/vrm_train_data.csv"
TEST_DATA_PATH="${CURRENT_DIR}/data/vrm_val_data.csv"
#EMBEDDING_PATH="${CURRENT_DIR}/data/GoogleNews-vectors-negative300.bin"
EMBEDDING_PATH="${CURRENT_DIR}/data/word2vec_from_glove.bin"
WORD_PATH="${CURRENT_DIR}/data/word.txt"
CHECKPOINT_PATH="${CURRENT_DIR}/trained_model/trained_model_VRM.pth" # put path for checkpoint
LOG_DIR="${CURRENT_DIR}/log"
CHCK_DIR="${CURRENT_DIR}/checkpoint"
OUTPUT_DIR="${CURRENT_DIR}/data"

python main.py \
    --command="test" \
    --net="BiLSTM-RAM" \
    --multiplier=1 \
    --dataset="VRM" \
    --data_path="${TEST_DATA_PATH}" \
    --embedding_path="${EMBEDDING_PATH}" \
    --num_workers=1 \
    --checkpoint_path="${CHECKPOINT_PATH}" \
    --checkpoint_folder="${CHCK_DIR}" \
    --output_folder="${OUTPUT_DIR}" \
    --testby_conv=1
