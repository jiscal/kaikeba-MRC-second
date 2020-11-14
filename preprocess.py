import numpy as np
import data_io as pio
import nltk

class Preprocessor:
    def __init__(self, datasets_fp, max_length=384,max_charlength=30, stride=128):
        self.datasets_fp = datasets_fp
        #一句中字符的最大长度
        self.max_length = max_length
        #句中单词的最大个数
        # self.max_wordlength=max_wordlength
        #单词中字符的最大长度
        self.max_charlength=max_charlength
        self.max_clen = 100
        self.max_qlen = 100
        self.stride = stride
        self.charset = set()
        self.build_charset()
        # self.build_GloveEmbedding()

    def build_GloveEmbedding(self,word_list,maxlen=120,embedding_dim=100):
        self.embedding_index={}
        with open("./data/glove.6B.100d.txt",'r',encoding='utf-8') as f:
            count=0
            for line in f.readlines():
                count+=1
                values=line.split()
                values = line.split()
                index = len(values) - 100
                if len(values) > (100 + 1):
                    word = ""  # 一个空的字符串。
                    for i in range(len(values) - 100):
                        word += values[i] + " "
                    word = word.strip()
                else:
                    word = values[0]
                coefs = np.asarray(values[index:], dtype='float32')
                self.embedding_index[word] = coefs
            embedding_matrix = np.zeros((maxlen, embedding_dim))
            for i, word in enumerate(word_list):
                value = self.embedding_index.get(word)
                if value is not None:
                    # print(len(value))
                    # 超过部分截取
                    if i < maxlen:
                        embedding_matrix[i] = value
        return  embedding_matrix


#每个字符构建字典
    def build_charset(self):
        for fp in self.datasets_fp:
            self.charset |= self.dataset_info(fp)

        self.charset = sorted(list(self.charset))
        self.charset = ['[PAD]', '[CLS]', '[SEP]'] + self.charset + ['[UNK]']
        idx = list(range(len(self.charset)))
        self.ch2id = dict(zip(self.charset, idx))
        self.id2ch = dict(zip(idx, self.charset))
        print(self.ch2id, self.id2ch)

    def dataset_info(self, inn):
        charset = set()
        dataset = pio.load(inn)

        for _, context, question, answer, _ in self.iter_cqa(dataset):
            charset |= set(context) | set(question) | set(answer)
            # self.max_clen = max(self.max_clen, len(context))
            # self.max_qlen = max(self.max_clen, len(question))

        return charset
#获取内容，问题，答案的
    def iter_cqa(self, dataset):
        for data in dataset['data']:
            for paragraph in data['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    qid = qa['id']
                    question = qa['question']
                    for answer in qa['answers']:
                        text = answer['text']
                        answer_start = answer['answer_start']
                        yield qid, context, question, text, answer_start

    def encode(self, context, question):
        question_encode = self.convert2id(question, begin=True, end=True)
        left_length = self.max_length - len(question_encode)
        context_encode = self.convert2id(context, maxlen=left_length, end=True)
        cq_encode = question_encode + context_encode

        assert len(cq_encode) == self.max_length

        return cq_encode

#把每一个句子分词然后加载Glove进行词向量矩阵，然后将词进行char级别的初始化字符向量矩阵
    def convertchar2id(self,sent, maxlen=120, begin=False, end=False):
        wordlist=nltk.word_tokenize(sent)
        #获取Glove词嵌入矩阵
        word_embedding_matrix=self.build_GloveEmbedding(wordlist,maxlen=maxlen)
        # print(word_embedding_matrix.shape)

        #存储对应单词中字符的表现
        char_dict={}
        count=0
        for i,word in enumerate(wordlist):
            # 初始化词中字符的矩阵
            char_embedding_matrix = np.zeros((self.max_charlength, 100))
            for j,char in enumerate(word):
                count+=1
                #随机初始化给字符正态的100维向量
                char_vector=np.random.normal(0,1,(1,100))
                # 超过部分截取
                if j<self.max_charlength:
                    char_embedding_matrix[j]=char_vector
            char_dict[i]=char_embedding_matrix
        return word_embedding_matrix,char_dict,count



#把每一个句子分成字符并进行id转换
    def convert2id(self, sent, maxlen=None, begin=False, end=False):
        ch = [ch for ch in sent]
        ch = ['[CLS]'] * begin + ch

        if maxlen is not None:
            ch = ch[:maxlen - 1 * end]
            ch += ['[SEP]'] * end
            ch += ['[PAD]'] * (maxlen - len(ch))
        else:
            ch += ['[SEP]'] * end

        ids = list(map(self.get_id, ch))

        return ids

    def get_id(self, ch):
        return self.ch2id.get(ch, self.ch2id['[UNK]'])
#c,q分别为一个对应长度的list
    def get_dataset(self, ds_fp):
        cs, qs, be = [], [], []
        for _, c, q, b, e in self.get_data(ds_fp):
            cs.append(c)
            qs.append(q)
            be.append((b, e))
        return map(np.array, (cs, qs, be))
    #字符和单词级别矩阵
    def get_chardataset(self, ds_fp):
        cs, qs, be = [], [], []
        #c,q分别是一个list，第一个是Glove词嵌入矩阵，第二个元素是一个字典key对应词的index，value是词的char初始化矩阵
        for _, c, q, b, e in self.get_chardata(ds_fp):
            print(type(c),len(c))
            cs.append(c)
            qs.append(q)
            be.append((b, e))
        return map(np.array, (cs, qs, be))

    def get_chardata(self, ds_fp):
        dataset = pio.load(ds_fp)
        for qid, context, question, text, answer_start in self.iter_cqa(dataset):
            # cids = self.get_sent_ids(context, self.max_clen)
            q_embedding_matrix, q_dict,_ = self.convertchar2id(question,self.max_qlen)
            c_embedding_matrix, c_dict,c_count= self.convertchar2id(context, self.max_clen)
            # qids = self.get_sent_ids(question, self.max_qlen)

            b, e = answer_start, answer_start + len(text)
            if e >= c_count:
                b = e = 0
            yield qid, [c_embedding_matrix, c_dict], [q_embedding_matrix, q_dict], b, e


#获取每一个内容，问题，答案并转换成id
    def get_data(self, ds_fp):
        dataset = pio.load(ds_fp)
        for qid, context, question, text, answer_start in self.iter_cqa(dataset):
            cids = self.get_sent_ids(context, self.max_clen)
            qids = self.get_sent_ids(question, self.max_qlen)
            b, e = answer_start, answer_start + len(text)
            if e >= len(cids):
                b = e = 0
            yield qid, cids, qids, b, e

    def get_sent_ids(self, sent, maxlen):
        return self.convert2id(sent, maxlen=maxlen, end=True)


if __name__ == '__main__':
    p = Preprocessor([
        './data/squad/train-v1.1.json',
        './data/squad/dev-v1.1.json',
        './data/squad/dev-v1.1.json'
    ])
    # train_c, train_q, train_y = p.get_dataset('./data/squad/train-v1.1.json')
    train_c, train_q, train_y = p.get_chardataset('./data/squad/train-v1.1.json')
    # print(train_c.shape, train_q.shape, train_y.shape)
    #print(p.encode('modern stone statue of Mary', 'To whom did the Virgin Mary '))
