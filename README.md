# How to start training
Under the experiments folder you can setup a new script with the training config as was done for the `clip_twi_testing.sh` script. Make sure that when you call python3 it is actually the python env you are expecting, simply using python instead of python3 is fine aslong as it is a version of python 3 of course. That should be everything for training CLIP!

Also go over the `clip_finetune_twi.py` file just to make sur that you are indeed loading the image datafrom where you expect it to be. 

NOTE! After a training run the model params are saved, so we can do evaluation on that later with top-k accuracy using the FAISS index built with the train data from that run. There is no need to setup any folder structure for the outputs of this script - it does that by itself.