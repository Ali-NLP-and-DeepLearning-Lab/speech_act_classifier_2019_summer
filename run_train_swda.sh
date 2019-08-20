CURRENT_DIR=$(pwd)
DATA_PATH="${CURRENT_DIR}/data/swda/train_data.csv"
VAL_DATA_PATH="${CURRENT_DIR}/data/swda/val_data.csv"
TEST_DATA_PATH="${CURRENT_DIR}/data/swda/test_data.csv"
#EMBEDDING_PATH="${CURRENT_DIR}/data/GoogleNews-vectors-negative300.bin"
EMBEDDING_PATH="${CURRENT_DIR}/data/word2vec_from_glove.bin"
LOG_DIR="${CURRENT_DIR}/log"
CHCK_DIR="${CURRENT_DIR}/checkpoint"
OUTPUT_DIR="${CURRENT_DIR}/data"
#RESUME_POINT="${CURRENT_DIR}/checkpoint/BiLSTM-PQICR_SWDA_POS_33/checkpoint_0.7964089623601875_20.pth"

python main.py \
     --command="train" \
     --net="BiLSTM-RAM" \
     --multiplier=1 \
     --optim="Adam" \
     --milestones="100" \
     --lr=0.01 \
     --dataset="SWDA" \
     --data_path="${DATA_PATH}" \
     --embedding_path="${EMBEDDING_PATH}" \
     --batch_size=128 \
     --num_epochs=100 \
     --num_workers=4 \
     --logdir="${LOG_DIR}" \
     --log_stride=25 \
     --checkpoint_folder="${CHCK_DIR}" \
     --checkpoint_stride=10 \
     --validation_epochs=0.066 \
     --val_data_path="${VAL_DATA_PATH}" \
     --use_conv_val=1 \
     --gamma=1 \

