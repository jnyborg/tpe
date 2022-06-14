#!/bin/bash

AT1='austria/33UVP/2017'
DK1='denmark/32VNH/2017'
FR1='france/30TXT/2017'
FR2='france/31TCJ/2017'

# Scenario 1
MODEL_NAME=pseltae_AT1+DK1+FR1
SOURCE="$AT1 $DK1 $FR1"
TARGET=$FR2

python train.py -e $MODEL_NAME --source $SOURCE --target $TARGET
python train.py -e $MODEL_NAME\_no_pos --source $SOURCE --target $TARGET --with_pos_enc=False
python train.py -e $MODEL_NAME\_shiftaug --source $SOURCE --target $TARGET --with_shift_aug=True
python train.py -e $MODEL_NAME\_tpe_concat --source $SOURCE --target $TARGET --with_gdd_extra=True --with_pos_enc=False
python train.py -e $MODEL_NAME\_tpe_sinusoid --source $SOURCE --target $TARGET --pos_type default --with_gdd_enc=True
python train.py -e $MODEL_NAME\_tpe_fourier --source $SOURCE --target $TARGET --pos_type fourier
python train.py -e $MODEL_NAME\_tpe_recurrent --source $SOURCE --target $TARGET --pos_type rnn

# Scenario 2
MODEL_NAME=pseltae_AT1+FR1+FR2
SOURCE="$AT1 $FR1 $FR2"
TARGET=$DK1

python train.py -e $MODEL_NAME --source $SOURCE --target $TARGET
python train.py -e $MODEL_NAME\_no_pos --source $SOURCE --target $TARGET --with_pos_enc=False
python train.py -e $MODEL_NAME\_shiftaug --source $SOURCE --target $TARGET --with_shift_aug=True
python train.py -e $MODEL_NAME\_tpe_concat --source $SOURCE --target $TARGET --with_gdd_extra=True --with_pos_enc=False
python train.py -e $MODEL_NAME\_tpe_sinusoid --source $SOURCE --target $TARGET --pos_type default --with_gdd_enc=True
python train.py -e $MODEL_NAME\_tpe_fourier --source $SOURCE --target $TARGET --pos_type fourier
python train.py -e $MODEL_NAME\_tpe_recurrent --source $SOURCE --target $TARGET --pos_type rnn


# Scenario 3
MODEL_NAME=pseltae_AT1+DK1+FR2
SOURCE="$AT1 $DK1 $FR2"
TARGET=$FR1

python train.py -e $MODEL_NAME --source $SOURCE --target $TARGET
python train.py -e $MODEL_NAME\_no_pos --source $SOURCE --target $TARGET --with_pos_enc=False
python train.py -e $MODEL_NAME\_shiftaug --source $SOURCE --target $TARGET --with_shift_aug=True
python train.py -e $MODEL_NAME\_tpe_concat --source $SOURCE --target $TARGET --with_gdd_extra=True --with_pos_enc=False
python train.py -e $MODEL_NAME\_tpe_sinusoid --source $SOURCE --target $TARGET --pos_type default --with_gdd_enc=True
python train.py -e $MODEL_NAME\_tpe_fourier --source $SOURCE --target $TARGET --pos_type fourier
python train.py -e $MODEL_NAME\_tpe_recurrent --source $SOURCE --target $TARGET --pos_type rnn

# Scenario 4
MODEL_NAME=pseltae_DK1+FR1+FR2
SOURCE="$DK1 $FR1 $FR2"
TARGET=$AT1

python train.py -e $MODEL_NAME --source $SOURCE --target $TARGET
python train.py -e $MODEL_NAME\_no_pos --source $SOURCE --target $TARGET --with_pos_enc=False
python train.py -e $MODEL_NAME\_shiftaug --source $SOURCE --target $TARGET --with_shift_aug=True
python train.py -e $MODEL_NAME\_tpe_concat --source $SOURCE --target $TARGET --with_gdd_extra=True --with_pos_enc=False
python train.py -e $MODEL_NAME\_tpe_sinusoid --source $SOURCE --target $TARGET --pos_type default --with_gdd_enc=True
python train.py -e $MODEL_NAME\_tpe_fourier --source $SOURCE --target $TARGET --pos_type fourier
python train.py -e $MODEL_NAME\_tpe_recurrent --source $SOURCE --target $TARGET --pos_type rnn
