import numpy as np
import torch
import torch.autograd
import config
import json
from model import Model
from transformers import AutoTokenizer
from collections import defaultdict, deque
import os

dis2idx = np.zeros((1000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9
PAD = '<pad>'
UNK = '<unk>'
SUC = '<suc>'


class W2NER(object):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Model(config=config)
        self.model.load_state_dict(torch.load(os.path.join(config.model_checkpoint, 'pytorch_model.bin')))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint)

        with open(os.path.join(config.model_checkpoint, 'id2label.json'), 'r', encoding='utf-8') as f:
            self.id2label = json.load(f)

    def process_bert(self, instance, tokenizer):

        tokens = [tokenizer.tokenize(word) for word in instance]
        pieces = [piece for pieces in tokens for piece in pieces]
        bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
        bert_inputs = np.array([tokenizer.cls_token_id] + bert_inputs + [tokenizer.sep_token_id])

        sent_length = len(instance)
        grid_labels = np.zeros((sent_length, sent_length), dtype=np.int)
        pieces2word = np.zeros((sent_length, len(bert_inputs)), dtype=np.bool)
        dist_inputs = np.zeros((sent_length, sent_length), dtype=np.int)
        grid_mask2d = np.ones((sent_length, sent_length), dtype=np.bool)

        if tokenizer is not None:
            start = 0
            for i, pieces in enumerate(tokens):
                if len(pieces) == 0:
                    continue
                pieces = list(range(start, start + len(pieces)))
                pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                start += len(pieces)

        for k in range(sent_length):
            dist_inputs[k, :] += k
            dist_inputs[:, k] -= k

        for i in range(sent_length):
            for j in range(sent_length):
                if dist_inputs[i, j] < 0:
                    dist_inputs[i, j] = dis2idx[-dist_inputs[i, j]] + 9
                else:
                    dist_inputs[i, j] = dis2idx[dist_inputs[i, j]]
        dist_inputs[dist_inputs == 0] = 19

        return torch.tensor([bert_inputs], dtype=torch.long, device=self.device), \
               torch.tensor([grid_mask2d], dtype=torch.long, device=self.device), \
               torch.tensor([pieces2word], dtype=torch.long, device=self.device), \
               torch.tensor([dist_inputs], dtype=torch.long, device=self.device), \
               torch.tensor([sent_length], dtype=torch.long, device=self.device)

    def convert_index_to_text(self, index, type):
        text = "-".join([str(i) for i in index])
        text = text + "-#-{}".format(type)
        return text

    def convert_text_to_index(self, text):
        index, type = text.split("-#-")
        index = [int(x) for x in index.split("-")]
        return index, int(type)

    def decode(self, outputs, length):
        class Node:
            def __init__(self):
                self.THW = []  # [(tail, type)]
                self.NNW = defaultdict(set)  # {(head,tail): {next_index}}

        ent_r, ent_p, ent_c = 0, 0, 0
        decode_entities = []
        q = deque()
        for instance, l in zip(outputs, length):
            predicts = []
            nodes = [Node() for _ in range(l)]
            for cur in reversed(range(l)):
                heads = []
                for pre in range(cur + 1):
                    # THW
                    if instance[cur, pre] > 1:
                        nodes[pre].THW.append((cur, instance[cur, pre]))
                        heads.append(pre)
                    # NNW
                    if pre < cur and instance[pre, cur] == 1:
                        # cur node
                        for head in heads:
                            nodes[pre].NNW[(head, cur)].add(cur)
                        # post nodes
                        for head, tail in nodes[cur].NNW.keys():
                            if tail >= cur and head <= pre:
                                nodes[pre].NNW[(head, tail)].add(cur)
                # entity
                for tail, type_id in nodes[cur].THW:
                    if cur == tail:
                        predicts.append(([cur], type_id))
                        continue
                    q.clear()
                    q.append([cur])
                    while len(q) > 0:
                        chains = q.pop()
                        for idx in nodes[chains[-1]].NNW[(cur, tail)]:
                            if idx == tail:
                                predicts.append((chains + [idx], type_id))
                            else:
                                q.append(chains + [idx])

            predicts = set([self.convert_index_to_text(x[0], x[1]) for x in predicts])
            decode_entities.append([self.convert_text_to_index(x) for x in predicts])
            ent_p += len(predicts)
        return ent_c, ent_p, ent_r, decode_entities

    def predict(self, sentence):
        """
        :param data: 输入样本
        :return: 返回实体
        """
        bert_inputs, grid_mask2d, pieces2word, dist_inputs, sent_length = self.process_bert(
            instance=sentence, tokenizer=self.tokenizer)

        with torch.no_grad():
            outputs = self.model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
            length = sent_length

            outputs = torch.argmax(outputs, -1)
            ent_c, ent_p, ent_r, decode_entities = self.decode(outputs.cpu().numpy(), length.cpu().numpy())

            sentences = {"sentence": sentence, "entity": []}
            for ent in decode_entities[0]:
                sentences["entity"].append({"text": ''.join([sentence[x] for x in ent[0]]),
                                            "type": self.id2label.get(ent[1], 'null')})

        return sentences


if __name__ == '__main__':
    w2ner = W2NER()
    sentence = ['喝', '奶', '时', '会', '咳', '嗽']
    print(w2ner.predict(sentence))
