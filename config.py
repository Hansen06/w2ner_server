#
import os

model_checkpoint = "./models/resume"
dist_emb_size = 20
type_emb_size = 20
lstm_hid_size = 512
conv_hid_size = 96
bert_hid_size = 768
biaffine_size = 512
ffnn_hid_size = 288
dilation = [1, 2, 3]
emb_dropout = 0.5
conv_dropout = 0.5
out_dropout = 0.33
use_bert_last_4_layers = True
label_num = 10

port = int(os.environ.get('PORT', '33136'))