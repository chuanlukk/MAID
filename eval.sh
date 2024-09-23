NAME="--name exp-name"
CLASSES="--num_classes 8"
MODEL="--model_path path/to/your/model_epoch_best.pth"
DATAROOT="--dataroot path/to/your/DMADataset"
python eval.py $DATAROOT $NAME $CLASSES $MODEL > ./eval-result-record.txt
