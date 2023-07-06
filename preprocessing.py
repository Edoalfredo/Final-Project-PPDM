from lib import*


def preprocess_text(text):
    text= text.lower()

    # remove tab, new line, and backslash
    text = text.replace('\\t', ' ').replace('\\n', ' ').replace('\\', '')
    # remove non ASCII (emoticon, Chinese word, etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", text).split())
    # remove incomplete URL
    text = text.replace("http://", " ").replace("https://", " ")
    # remove numbers
    text = re.sub(r"\d+", "", text)
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # remove leading and trailing whitespace
    text = text.strip()
    # remove multiple whitespace into single whitespace
    text = re.sub('\s+', ' ', text)
    # remove single character
    text = re.sub(r"\b[a-zA-Z]\b", "", text)
    # tokenize words
    tokens = word_tokenize(text.lower())
    # remove stopwords
    stopword_list = set(stopwords.words('indonesian'))
    tokens = [word for word in tokens if word not in stopword_list]
    return tokens

normalizad_word = pd.read_excel("normalisasi.xlsx")

normalizad_word_dict = {}

for index, row in normalizad_word.iterrows():
    if row[0] not in normalizad_word_dict:
        normalizad_word_dict[row[0]] = row[1] 

def normalized_term(document):
    return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]