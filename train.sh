# base
NAME="--name exp-name"
BS="--batch_size 256"
SPLIT="--train_split train --val_split val"
DATAROOT="--dataroot path/to/your/DMADataset"
CP="--checkpoints_dir path/to/your/checkpoints_dir"
CLASSES="--num_classes 8"
python train.py $DATAROOT $SPLIT $NAME $CLASSES $BS $CP > ./train-result-record.txt
