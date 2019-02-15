import re
import torch
from torch import nn
import os
from ELMoForManyLangs import elmo
import numpy as np
from jexus import Clock, History
import math
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import argparse
from pypinyin import pinyin, lazy_pinyin, Style

def f2h(s):
    # return re.sub(r"( |　)+", " ", s).strip()
    s = list(s)
    for i in range(len(s)):
        num = ord(s[i])
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        s[i] = chr(num).translate(str.maketrans('﹕﹐﹑。﹔﹖﹗﹘　', ':,、。;?!- '))
    return re.sub(r"( |　)+", " ", "".join(s)).strip()
    
def split_valid_list(X, v_size=0.05, rand=True):
    if rand == True:
        shuffle(X)
    t_size = math.floor(len(X) * (1 - v_size))
    X_v = X[t_size:]
    X = X[:t_size]
    return X, X_v

def sort_list(li, piv=2,unsort_ind=None):
    ind = []
    if unsort_ind == None:
        ind = sorted(range(len(li[piv])), key=(lambda k: li[piv][k]))
    else:
        ind = unsort_ind
    for i in range(len(li)):
        li[i] = [li[i][j] for j in ind]
    return li, ind

def sort_numpy(li, piv=2,unsort=False):
    ind = np.argsort(-li[piv] if not unsort else li[piv], axis=0)
    for i in range(len(li)):
        if type(li[i]).__module__ == np.__name__ or type(li[i]).__module__ == torch.__name__:
            li[i] = li[i][ind]
        else:
            li[i] = [li[i][j] for j in ind]
    return li, ind

def sort_torch(li, piv=2,unsort=False):
    li[piv], ind = torch.sort(li[piv], dim=0, descending=(not unsort))
    for i in range(len(li)):
        if i == piv:
            continue
        else:
            li[i] = li[i][ind]
    return li, ind

def sort_by(li, piv=2, unsort=False):
    if type(li[piv]).__module__ == np.__name__:
        return sort_numpy(li, piv, unsort)
    elif type(li[piv]).__module__ == torch.__name__:
        return sort_torch(li, piv, unsort)
    else:
        return sort_list(li, piv, unsort)

class Embedder():
    def __init__(self, seq_len=0, use_cuda=True, device=None):
        self.embedder = elmo.Embedder(model_dir="zhuyin.model", batch_size=512, use_cuda=use_cuda)
        self.seq_len = seq_len
        self.device = device
        if self.device != None:
            self.embedder.model.to(self.device)
        self.bos_vec, self.eos_vec, self.pad, self.oov = self.embedder.sents2elmo([["<bos>","<eos>","<pad>","<oov>"]], output_layer=0)[0]

    def __call__(self, sents, max_len=0, with_bos_eos=True, layer=-1, pad_matters=False):
        seq_lens = np.array([len(x) for x in sents], dtype=np.int64)
        sents = [[self.sub_unk(x) for x in sent] for sent in sents]
        if max_len != 0:
            pass
        elif self.seq_len != 0:
            max_len = self.seq_len
        else:
            max_len = seq_lens.max()
        emb_list = self.embedder.sents2elmo(sents, output_layer=layer)
        if not with_bos_eos:
            for i in range(len(emb_list)):
                if max_len - seq_lens[i] > 0:
                    if pad_matters:
                        emb_list[i] = np.concatenate([emb_list[i], np.tile(self.pad,[max_len - seq_lens[i],1])], axis=0)
                    else:
                        emb_list[i] = np.concatenate([emb_list[i], np.zeros((max_len - seq_lens[i], emb_list[i].shape[1]))])
                else:
                    emb_list[i] = emb_list[i][:max_len]
        elif with_bos_eos:
            for i in range(len(emb_list)):
                if max_len - seq_lens[i] > 0:
                    if pad_matters:
                        emb_list[i] = np.concatenate([
                            self.bos_vec[np.newaxis],
                            emb_list[i],
                            self.eos_vec[np.newaxis],
                            np.tile(self.pad, [max_len - seq_lens[i], 1])], axis=0)
                    else:
                        emb_list[i] = np.concatenate([
                            self.bos_vec[np.newaxis],
                            emb_list[i],
                            self.eos_vec[np.newaxis],
                            np.zeros((max_len - seq_lens[i], emb_list[i].shape[1]))], axis=0)
                else:
                    emb_list[i] = np.concatenate([self.bos_vec[np.newaxis], emb_list[i][:max_len],self.eos_vec[np.newaxis]], axis=0)
        embedded = np.array(emb_list, dtype=np.float32)
        seq_lens = seq_lens+2 if with_bos_eos else seq_lens
        return embedded, seq_lens

    def sub_unk(self, e):
        e = e.replace('，',',')
        e = e.replace('：',':')
        e = e.replace('；',';')
        e = e.replace('？','?')
        e = e.replace('！', '!')
        return e

def load_vocab(limit=10000):
    __file__ = '.'
    idx2word = ["<pad>","<unk>"] + list(np.load(os.path.join(os.path.dirname(__file__),"CharEmb/idx2word.npy")))[:limit-2]
    word2idx = dict([(word, i) for i, word in enumerate(idx2word)])
    return idx2word, word2idx

class Utils():
    def __init__(self, zhuyin_data_path="./zhuyin_corpus.txt",
    char_data_path="./char_corpus.txt",
    test_zhuyin_data_path="./test.zhuyin",
    test_char_data_path="./test.char",
    batch_size = 32, elmo_device=None, max_len=20):
        self.zhuyin_data_path = zhuyin_data_path
        self.test_zhuyin_data_path = test_zhuyin_data_path
        self.zhuyin_line_num = int(os.popen("wc -l %s"%self.zhuyin_data_path).read().split(' ')[0])
        self.test_zhuyin_line_num = int(os.popen("wc -l %s"%self.test_zhuyin_data_path).read().split(' ')[0])
        self.char_data_path = char_data_path
        self.test_char_data_path = test_char_data_path
        self.char_line_num = int(os.popen("wc -l %s"%self.char_data_path).read().split(' ')[0])
        self.test_char_line_num = int(os.popen("wc -l %s"%self.test_char_data_path).read().split(' ')[0])
        self.elmo = Embedder(device=elmo_device)
        self.batch_size = batch_size
        self.max_len = max_len
        self.train_step_num = math.floor(self.zhuyin_line_num / batch_size)
        self.test_step_num = math.floor(self.test_char_line_num / batch_size)
        self.idx2word, self.word2idx = load_vocab()

    def sent2list(self, sent, zy=True):
        sent = f2h(sent)
        word_list = re.split(r"[\s|\u3000]+", sent.strip())
        return [x if x in self.word2idx else "<unk>" for x in word_list] if not zy else word_list

    def data_generator(self):
        file_x = open(self.zhuyin_data_path)
        file_y = open(self.char_data_path)
        sents = [[],[]]
        for sent_x, sent_y in zip(file_x, file_y):
            if len(sent_x.strip()) == 0 or len(sent_y.strip()) == 0:
                continue
            word_list_x = self.sent2list(sent_x)
            word_list_y = self.sent2list(sent_y, zy=False)
            if len(word_list_x) > self.max_len or len(word_list_y) > self.max_len:
                continue
            sents[0].append(word_list_x)
            sents[1].append(word_list_y)
            if len(sents[0]) == self.batch_size:
                yield sents
                sents = [[], []]
        if len(sents[0])!=0:
            yield sents

    def test_data_generator(self):
        file_x = open(self.test_zhuyin_data_path)
        file_y = open(self.test_char_data_path)
        sents = [[],[]]
        for sent_x, sent_y in zip(file_x, file_y):
            if len(sent_x.strip()) == 0 or len(sent_y.strip()) == 0:
                continue
            word_list_x = self.sent2list(sent_x)
            word_list_y = self.sent2list(sent_y, zy=False)
            sents[0].append(word_list_x)
            sents[1].append(word_list_y)
            if len(sents[0]) == self.batch_size:
                yield sents
                sents = [[], []]
        if len(sents[0])!=0:
            yield sents

    def sents2idx(self, raw_sents, len_fixed=True):
        max_len = self.max_len
        if not len_fixed:
            max_len = max([len(x) for x in raw_sents])
        ret = np.zeros((len(raw_sents), max_len), dtype=int)
        for i in range(len(raw_sents)):
            for j in range(len(raw_sents[i])):
                ret[i][j] = self.word2idx[raw_sents[i][j]]
        return ret


class Zhuyin2Char(nn.Module):
    def __init__(self,
        batch_size=32,
        device="cuda:1",
        elmo_device="cuda:0",
        hidden_size=300,
        input_size=1024,
        vocab_size=10000,
            n_layers=3,
            utils=None,
            dropout=0.33,
            save_path=""):
        super(Zhuyin2Char, self).__init__()
        self.utils = utils
        self.device = device if torch.cuda.is_available() else "cpu"
        self.gru = nn.LSTM(input_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout),
                          bidirectional=True,
                          batch_first=True)
        self.fc1 = nn.Linear(2*hidden_size, vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.Adam(self.parameters())
        self.save_path = save_path

    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = torch.from_numpy(input_seq).to(self.device)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        outputs, hidden = self.gru(packed, hidden) # output: (seq_len, batch, hidden*n_dir)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        pred_prob = self.fc1(outputs)#nn.Softmax(dim=-1)(self.fc1(outputs))
        return pred_prob

    def demo_sent(self, sents, unsort=False):
        embedded, seq_lens = self.utils.elmo(sents)
        (embedded, seq_lens, sents), ind = sort_by([embedded, seq_lens, sents], piv=1)
        pred = self.forward(embedded, seq_lens)
        ans = torch.argmax(pred, dim=-1).cpu().numpy()[:, 1:]
        new_sents = []
        for i, sent in enumerate(sents):
            new_sent = []
            for j, char in enumerate(sent):
                if j >= seq_lens[i]:
                    break
                pred = self.utils.idx2word[ans[i,j]]
                new_sent.append(pred if pred!="<unk>" else "{"+sents[i][j]+"}")
            new_sents.append(new_sent)
        if unsort:
            (new_sents, ind), _ind = sort_by([new_sents, ind], piv=1, unsort=True)
        return new_sents

    def demo(self):
        with torch.no_grad():
            self.eval()
            for step, batch_x in enumerate(self.utils.data_generator(mode="test")):
                sents = batch_x[0]
                result = self.demo_sent(sents, unsort=True)
                for i in result:
                    print(' '.join(i))

    def test_and_write_file(self, filename, actual_name):
        fw = open(filename, 'w')
        with torch.no_grad():
            self.eval()
            # ct = Clock(self.utils.test_step_num)
            for step, batch_x in enumerate(self.utils.data_generator(mode="test", write_actual_data=True, actual_name=actual_name)):
                sents = batch_x[0]
                result = self.demo_sent(sents, unsort=True)
                fw.writelines([' '.join(x) + '\n' for x in result])
                # ct.flush()

    def live_demo(self):
        with torch.no_grad():
            self.eval()
            while True:
                sent = input("sent> ")
                if len(sent.strip()) == 0:
                    continue
                # sent = sent.split(' ')
                sent = [x[0] for x in pinyin(sent, style=Style.BOPOMOFO)]
                result = self.demo_sent([sent])
                print("--")
                print(re.sub(r"[\s|\u3000]+", " ",' '.join(result[0]).strip())+'\n')


    def test_corpus(self, return_value=False):
        with torch.no_grad():
            self.eval()
            info = {"acc": 0}
            total = 0
            ct = Clock(self.utils.test_step_num)
            for step, (batch_x, batch_y) in enumerate(self.utils.test_data_generator()):
                elmo_x, x_lens = self.utils.elmo(batch_x)
                (elmo_x, x_lens, batch_y), _ind = sort_by([elmo_x, x_lens, batch_y], piv=1)
                label = self.utils.sents2idx(batch_y, len_fixed=False)
                target = torch.from_numpy(label).to(self.device) if self.device!="cpu" else torch.from_numpy(label)
                pred = self.forward(elmo_x, x_lens)[:, 1:-1]
                ans = torch.argmax(pred, dim=-1).cpu().numpy()
                idx = (target > 0).nonzero()
                acc = accuracy_score(label[idx[:,0], idx[:,1]], ans[idx[:,0], idx[:,1]])
                info["acc"]+=acc
                total += 1
                target.cpu()
                ct.flush()
            self.train()
            print("test_acc:", info["acc"] / total)
            if return_value:
                return info["acc"] / total
            else:
                return 

    def train_model(self, num_epochs=1, step_to_save_model=1000, check_point=False):
        self.to(self.device)
        self.train()
        max_acc = 0.0
        for epoch in range(num_epochs):
            ct = Clock(self.utils.train_step_num, title="Epoch(%d/%d)"%(epoch+1, num_epochs))
            His_loss = History(title="Loss", xlabel="step", ylabel="loss",
            item_name=["train_loss"])
            His_acc = History(title="Acc", xlabel="step", ylabel="accuracy",
            item_name=["train_acc"])
            for step, (batch_x, batch_y) in enumerate(self.utils.data_generator()):
                elmo_x, x_lens = self.utils.elmo(batch_x, max_len=self.utils.max_len)
                (elmo_x, x_lens, batch_y), _ind = sort_by([elmo_x, x_lens, batch_y], piv=1)
                label = self.utils.sents2idx(batch_y)
                target = torch.from_numpy(label).to(self.device) if self.device!="cpu" else torch.from_numpy(label)
                pred = self.forward(elmo_x, x_lens)[:, 1:-1]
                if pred.shape[1] < target.shape[1]:
                    p1d = (pred.shape[0], target.shape[1] - pred.shape[1], pred.shape[2])
                    to_pad = torch.zeros(*p1d).to(self.device)
                    pred = torch.cat([pred, to_pad], dim=1)
                try:
                    loss = self.criterion(pred.transpose(1, 2), target)
                except:
                    print(batch_x)
                    print(batch_y)
                    sys.exit()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                ans = torch.argmax(pred, dim=-1).cpu().numpy()
                idx = (target>0).nonzero()
                acc = accuracy_score(label[idx[:,0], idx[:,1]], ans[idx[:,0], idx[:,1]])
                info_dict = {"loss": loss, "ppl":math.exp(loss),"acc":acc}
                ct.flush(info=info_dict)
                His_loss.append_history(0, (step, loss))
                His_acc.append_history(0, (step, acc))
                target.cpu()
                if (step + 1) % step_to_save_model == 0:
                    torch.save(self.state_dict(), os.path.join(self.save_path,'model.ckpt'))
                    His_loss.plot(os.path.join(self.save_path,"loss_plot"))
                    His_acc.plot(os.path.join(self.save_path,"acc_plot"))
            # acc, f1 = self.test_corpus(return_value=True)
            if check_point:
                if acc > max_acc:
                    path = os.path.join(self.save_path,'model.ckpt')
                    print("Checkpoint: acc grow from %4f to %4f, save model to %s"%(max_acc, acc, path))
                    torch.save(self.state_dict(), path)
                    max_acc = acc
                else:
                    print("Checkpoint: acc not grow from %4f to %4f, model not save."%(max_acc, acc))
            else:
                path = os.path.join(self.save_path,'model.ckpt')
                torch.save(self.state_dict(), path)
        His_loss.plot(os.path.join(self.save_path,"Loss_"+self.utils.zhuyin_data_path.split('/')[-1]+"_%d"%num_epochs))
        His_acc.plot(os.path.join(self.save_path,"Acc_"+self.utils.zhuyin_data_path.split('/')[-1]+"_%d"%num_epochs))

    def load_model(self, filename='model.ckpt', device=None):
        if device == None:
            device = self.device
        self.load_state_dict(torch.load(os.path.join(self.save_path,filename), map_location=device))
        print("model.ckpt load!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="execute mode")
    parser.add_argument("-filename", default=None, required=False, help="test filename")
    parser.add_argument("-load_model", default=True, required=False, help="test filename")
    parser.add_argument("-model_name", default="model.ckpt", required=False, help="test filename")
    parser.add_argument("-save_path", default="", required=False, help="test filename")
    parser.add_argument("-train_file", default="small.zhuyin", required=False, help="test filename")
    parser.add_argument("-test_file", default="small.char", required=False, help="test filename")
    parser.add_argument("-eval_filez", default="small.zhuyin", required=False, help="test filename")
    parser.add_argument("-eval_filec", default="small.char", required=False, help="test filename")
    parser.add_argument("-epoch", default=1, required=False, help="test filename")
    parser.add_argument("-max_len", default=20, required=False, help="test filename")
    args = parser.parse_args()
    model = Zhuyin2Char(device="cuda:0", utils=Utils(zhuyin_data_path=args.train_file,
    char_data_path=args.test_file, test_char_data_path=args.eval_filec, test_zhuyin_data_path=args.eval_filez, batch_size=32, elmo_device="cuda:0", max_len=int(args.max_len)), save_path=args.save_path)
    model.to(model.device)
    if args.load_model:
        model.load_model(filename=args.model_name)
    if args.mode == "train":
        model.train_model(num_epochs=int(args.epoch))
        print("========= Testing =========")
        model.load_model()
        model.test_corpus()
    if args.mode == "good":
        model.train_model(num_epochs=int(args.epoch))
        new_utils = Utils(zhuyin_data_path="zhuyin.comma",
            char_data_path="char.comma",
            test_char_data_path=args.eval_filec,
            test_zhuyin_data_path=args.eval_filez,
            batch_size=32,
            elmo_device="cuda:0",
            max_len=40, save_path=args.save_path)
        model.utils = new_utils
        model.train_model(num_epochs=10)
        print("========= Testing =========")
        model.load_model()
        model.test_corpus()
    if args.mode == "test":
        model.test_corpus()
    if args.mode == "demo":
        model.demo()
    if args.mode == "live":
        model.live_demo()
    if args.mode == "output":
        model.output_train_file(args.filename)
    if args.mode == "write":
        model.test_and_write_file(filename=args.filename, actual_name=args.actual_name)
    if args.mode == "check_train":
        model.train_model()
        model.load_model()
        model.test_corpus()
