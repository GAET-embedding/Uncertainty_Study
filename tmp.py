import pandas as pd


def generate_indexifyer(self):
    def indexify(lst_text):
        indices = []
        for word in lst_text:
            if word in self.word_to_index:
                indices.append(self.word_to_index[word])
            else:
                indices.append(self.word_to_index['__UNK__'])
        return indices
    return indexify


def _tokenize(text):
    # return [x.lower() for x in nltk.word_tokenize(text)]
    return [x.lower() for x in text.split()]


def preprocess(path_file):
    df = pd.read_csv(path_file, delimiter='\t')
    df['body'] = df['body'].apply(_tokenize)
    old_samples = df['body'].copy()
    df['body'] = df['body'].apply(generate_indexifyer())
    samples = df.values.tolist()