import faiss
import os
import json
import time
from contextlib import contextmanager
from copy import deepcopy
import numpy as np

from contextlib import contextmanager
import time

@contextmanager
def timer(silence=False):
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    if not silence:
        print(f"Elapsed time: {round(elapsed_time * 1000,3)} ms")

def knowledge_tagging(sentence,split=False):
    import re
    pattern = '宝马|奔驰|奥迪|特斯拉|保时捷|奇瑞汽车|比亚迪|华为|吉利|智己|纪梵希|宁德时代|智界|苹果|大众|五菱宏光|阿维塔|问界|小米|理想|长安|奇瑞|小鹏|沃尔沃|长安汽车|戴姆勒|高合|江汽集团|腾势|ZEEKR极氪|赛力斯|丰田|福特|华人运通|奇瑞|BYD|Huawei|Geely|ZHiji|Givenchy|CATL|ZhiJie|Apple|Volkswagen|Wuling\\ Hongguang|Avita|AITO|Xiaomi|Li\\ Xiang|Changan|Chery|Xpeng|Volvo|Changan\\ Automobile|Daimler|HiPhi|Jiang\\ Automotive\\ Group|Denza|ZEEKR|Seres|Toyota|Ford|Human\\ Horizons|bmw|audi|tesla|mercedes-benz'

    nio_pattern = 'ET9|ET7|ET5T|ET5|ES8|ES6|ES7|EC7|EC6'

    pattern = pattern.lower()
    nio_pattern = nio_pattern
    mapping = {'奇瑞': '奇瑞汽车',
    'byd': '比亚迪',
    'huawei': '华为',
    'geely': '吉利',
    'zhiji': '智己',
    'givenchy': '纪梵希',
    'catl': '宁德时代',
    'zhijie': '智界',
    'apple': '苹果',
    'volkswagen': '大众',
    'wuling hongguang': '五菱宏光',
    'avita': '阿维塔',
    'aito': '问界',
    'xiaomi': '小米',
    'li xiang': '理想',
    'changan': '长安',
    'chery': '奇瑞',
    'xpeng': '小鹏',
    'volvo': '沃尔沃',
    'changan automobile': '长安汽车',
    'daimler': '戴姆勒',
    'hiphi': '高合',
    'jiang automotive group': '江汽集团',
    'denza': '腾势',
    'zeekr': 'ZEEKR极氪',
    'seres': '赛力斯',
    'toyota': '丰田',
    'ford': '福特',
    'human horizons': '华人运通',
    'bmw': '宝马',
    'audi': '奥迪',
    'mercedes-benz': '奔驰',
    'tesla': '特斯拉',}

    result = {}


    car_types = list(set(re.findall(pattern, (sentence).lower())))
    car_types = list(set([mapping.get(x, x) for x in car_types]))
    car_types = list(set(car_types))
    result['竞品'] = car_types
        
    lower_nio_pattern = nio_pattern.lower()
    nio_types = list(set(re.findall(lower_nio_pattern, (sentence).lower())))
    nio_types = [i.upper() for i in list(set(nio_types))]
    nio_types = [i for i in nio_types if i in nio_pattern.split('|')]
    nio_types = list(set(nio_types))
    result['NIO'] = nio_types

    if not split:
        result = (set(car_types + nio_types))
    else:
        pass
    return result


class FaissSentenceIndexer:
    def __init__(self,sentence_embeddings=None, sentence_list=[], keyword_set_list = [], d=1024, index_folder=None, silence=False):
        self.silence = silence
        with timer(silence=self.silence):
            if index_folder is not None:
                print("Loading FaissSentenceIndexer from", index_folder)
                self.index = faiss.read_index(os.path.join(index_folder, 'index'))
                with open(os.path.join(index_folder, 'sentence_list.json'), 'r') as f:
                    self.sentence_list = json.load(f)
                self.d = self.index.d
                print("index dimension:", self.d)
            else:
                self.d = d
                self.index = faiss.IndexFlatIP(d)

                if sentence_embeddings is not None:
                    if sentence_embeddings.shape[1] != d:
                        raise ValueError("sentence_embeddings dimension mismatch")
                    if sentence_embeddings.shape[0] != len(sentence_list):
                        raise ValueError("sentence_embeddings number mismatch with sentence_list")
                    if sentence_embeddings.shape[0] != len(keyword_set_list) and len(keyword_set_list) > 0:
                        raise ValueError("sentence_embeddings number mismatch with keyword_set_list")

                if sentence_embeddings is not None:
                    faiss.normalize_L2(sentence_embeddings)
                    self.index.add(sentence_embeddings)
                self.sentence_list = deepcopy(sentence_list)
                if len(keyword_set_list) == 0:
                    keyword_set_list = [set() for _ in range(len(sentence_list))]
                self.keyword_set_list = deepcopy(keyword_set_list)
    
    def keyword_match(self, keyword_set, index):
        output = []
        nio_only_count = 0
        other_only_count = 0
        for idx in index:
            i = self.keyword_set_list[idx]
            # print(type(i))
            # print(keyword_set)
            # 蔚来车型加分
            search1 = keyword_set & i
            search1_score = len(search1)
            # print(search1)
            nio_search2 = search1 & {'EC6', 'EC7', 'ES6', 'ES7', 'ES8', 'ET5', 'ET5T', 'ET7', 'ET9'}
            not_nio_keyword = search1 - {'EC6', 'EC7', 'ES6', 'ES7', 'ES8', 'ET5', 'ET5T', 'ET7', 'ET9'}
            # print(nio_search2)
            if len(nio_search2) > 0 and len(not_nio_keyword) == 0 and nio_only_count<1:
                nio_add_score = 0.5
                nio_only_count += 1
                # print('NIO 0.5')
            else: 
                nio_add_score = 0 
            
            if len(not_nio_keyword) > 0 and len(nio_search2) == 0 and other_only_count<1:
                other_add_score = 0.5
                other_only_count += 1
            else:
                other_add_score = 0

            score = search1_score + nio_add_score + other_add_score
            output.append(score)
        return output

    def search(self, sentence_embeddings, keyword = [], k=3):
        with timer(silence=self.silence):
            if len(sentence_embeddings.shape) != 2:
                raise ValueError("input should be 2-dim numpy array")

            faiss.normalize_L2(sentence_embeddings)

            

            if sentence_embeddings.shape[0] == 1:
                if len(keyword) > 0:
                    D, I = self.index.search(sentence_embeddings, len(self.sentence_list))
                    if isinstance(keyword,list):
                        keyword = set(keyword)
                    elif isinstance(keyword,str):
                        keyword = set([keyword,])
                    elif isinstance(keyword,set):
                        pass
                    else:
                        raise ValueError("keyword should be list or str or set") 
                    similarity = D[0].tolist()
                    index = I[0].tolist()
                    # 按照index的顺序输出
                    keyword_match = self.keyword_match(keyword,index)

                    final_output = list(zip(similarity,index,keyword_match))
                    final_output.sort(key=lambda x: (-x[2],-x[0]))
                    return [[self.keyword_set_list[s[1]]&keyword for s in final_output[:k]]], final_output, [[self.sentence_list[s[1]] for s in final_output[:k]]], keyword_match               
                else:
                    pass 
            else:
                
                if len(keyword) > 0:
                    print(' key word search for multi sentence do not support now !')
                else:
                    pass
            
            D, I = self.index.search(sentence_embeddings, k)
            return D, I, [[self.sentence_list[i] for i in I_row] for I_row in I]

    def add(self, sentence_embeddings, sentence_list):
        with timer(silence=self.silence):
            if len(sentence_embeddings.shape) != 2:
                raise ValueError("input should be 2-dim numpy array")

            if not isinstance(sentence_list, list):
                raise ValueError("sentence should be a list")

            if sentence_embeddings.shape[0] != len(sentence_list):
                raise ValueError("sentence number mismatch")

            if sentence_embeddings.shape[1] != self.d:
                raise ValueError("embedding dimension mismatch")

            faiss.normalize_L2(sentence_embeddings)
            self.index.add(sentence_embeddings)
            self.sentence_list += deepcopy(sentence_list)
    
    def save(self, index_folder='index_flat_ip'):
        with timer(silence=self.silence):
            if not os.path.exists(index_folder):
                os.makedirs(index_folder)
            faiss.write_index(self.index, os.path.join(index_folder, 'index'))
            with open(os.path.join(index_folder, 'sentence_list.json'), 'w') as f:
                json.dump(self.sentence_list, f)


if __name__ == '__main__':
    import numpy as np
    # import os
    # import json
    # os.environ['HTTP_PROXY'] = 'http://10.22.112.37:8080'
    # os.environ['HTTPS_PROXY'] = 'http://10.22.112.37:8080'

    # from sentence_transformers import SentenceTransformer
    # model = SentenceTransformer('thenlper/gte-large-zh')
    d = 1024

    import json
    with open('example_data.txt', 'r') as f:
        sentences = json.load(f)
    
    embeddings = np.random.rand(30000, d).astype('float32')

    sentence_indexer = FaissSentenceIndexer(embeddings, sentences, d=d)

    tag = '我要吃饭'
    embedding = np.random.rand(1, d).astype('float32')
    # embedding = model.encode([tag], normalize_embeddings=True, show_progress_bar=True, device='cuda:0')

    sentence_indexer.search(embedding, k=29999)

    sentence_indexer.search(embedding, k=10000)

    sentence_indexer.search(embedding, k=1000)

    sentence_indexer.add(embedding, [tag])

    print(sentence_indexer.search(embedding, k=3))

    sentence_indexer.save('index_flat_ip')

    sentence_indexer2 = FaissSentenceIndexer(index_folder='index_flat_ip')

    print(sentence_indexer2.search(embedding, k=3))
