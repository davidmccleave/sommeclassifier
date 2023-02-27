import joblib
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tag import pos_tag

classes = ['Pinot Noir', 'Chardonnay', 'Cabernet Sauvignon', 'Red Blend', 'Bordeaux-style Red Blend', 'Riesling', 'Sauvignon Blanc',
 'Syrah', 'Rosé', 'Merlot', 'Zinfandel', 'Malbec', 'Sangiovese', 'Nebbiolo', 'Portuguese Red', 'White Blend', 'Sparkling Blend',
 'Tempranillo', 'Rhône-style Red Blend', 'Pinot Gris', 'Cabernet Franc', 'Champagne Blend', 'Grüner Veltliner', 'Pinot Grigio',
 'Portuguese White', 'Viognier', 'Gewürztraminer', 'Gamay', 'Shiraz', 'Petite Sirah', 'Bordeaux-style White Blend', 'Grenache',
 'Barbera', 'Glera', 'Sangiovese Grosso', 'Tempranillo Blend', 'Carmenère', 'Chenin Blanc'
]

# ------------------------------ Entry Points ------------------------------

def get_trained_classifier():
    print('Loading classifier ...')
    return joblib.load('nb-classifier.joblib')

def return_recommendations(description, classifier):
    print('return_recommendations...')
    class_probabilities = list()
    converted_input = convert_input(description)
    for k in classes:
        prob = classifier.prob_classify(converted_input).prob(k)
        probability = round(100 * prob, 2)
        item = [k, probability]
        class_probabilities.append(item)
    class_probabilities.sort(key=lambda x: x[1], reverse=True)
    top3 = class_probabilities[:3]
    return top3

# ----------------------- User Description Processing -----------------------

def convert_input(text):
    lemmatizer = WordNetLemmatizer()
    # Tokenize
    tokens = lemmatize_tokens(clean_tokens(text.split()), lemmatizer)
    return {token: True for token in tokens}

def clean_tokens(tokens):
    cleaned = []
    for token in tokens:
        if token.startswith('@'):
            continue
        cleaned.append(token.lower().replace(',', '').replace('.', ''))
    return cleaned

def lemmatize_tokens(tokens, lemmatizer):
    tagged_tokens = pos_tag(tokens)
    return [lemmatize(token[0], token[1], lemmatizer) for token in tagged_tokens]

def lemmatize(word, tag, lemmatizer):
    tag = get_wordnet_tag(tag)
    if tag != '':
        return lemmatizer.lemmatize(word, tag)
    return word

def get_wordnet_tag(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''
