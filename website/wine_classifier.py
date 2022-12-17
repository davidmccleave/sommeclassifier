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


# import pandas as pd
# from sklearn.model_selection import train_test_split
# import nltk
# from nltk import NaiveBayesClassifier

# nltk.data.path.append('./nltk_data/')

# frequency_threshold = 500
# reviews = None
# ready = False

# def get_trained_classifier():
    # global df
    # drop_cols = ['designation', 'title', 'region_1', 'region_2', 'taster_name', 'taster_twitter_handle']
    # wine_df = pd.read_csv('data/winemag-data-130k-v2.csv', index_col=0).drop(columns=drop_cols)
    # wine_df = filter_df(wine_df)
    # set_reviews(wine_df)
    # df = wine_df
    # print('Loading classifier...')
    # return process(df)
    # classifier = None
    # with open('naivebayes.pickle', 'rb') as classifier_f:
    #     classifier = pickle.load(classifier_f)
    #     classifier_f.close()
    # print('Complete')
    # return classifier


# def filter_df(df):
#     # global classes
#     df = df.loc[df['variety'].isin(classes)]
#     return df

# def set_reviews(df):
#     global reviews
#     reviews = df['description'].to_list()
#     print('Tokenizing and cleaning...')
#     # Tokenize
#     for i in range(len(reviews)):
#         reviews[i] = reviews[i].split()
#     # Clean
#     reviews = [clean_tokens(review) for review in reviews]

# def process(df):
#     global ready
#     lemmatizer = WordNetLemmatizer()
    
#     print('Lemmatizing tokens...')
#     lemmatized_reviews = [lemmatize_tokens(review, lemmatizer) for review in reviews]

#     def prepare_tokens(tokens, index):
#         return ({token: True for token in tokens}, df.iloc[index]['variety'])

#     # Training
#     prepped_reviews = [prepare_tokens(review, index) for index, review in enumerate(lemmatized_reviews)]
#     train_data, test_data = train_test_split(prepped_reviews, shuffle=True, test_size=0.3, random_state=0)
#     print('Beginning classifier training...')
#     classifier = NaiveBayesClassifier.train(train_data)
#     print('Done!')
#     ready = True
#     return classifier

#     # selections = []
#     # for i in range(3):
#     #     selection_df = df[df['variety'] == top3[i][0]]
#     #     selection_df = selection_df.sort_values(by=['points', 'price'], ascending=False)
#     #     selections.append(selection_df.iloc[0])
