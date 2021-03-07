from pathlib import Path
from word2vec import get_word2vec_instance
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

from gensim.utils import simple_preprocess
from sklearn.neighbors import NearestNeighbors


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    out_idx = []
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        out_idx.append((int(last), int(last + avg)))
        last += avg

    return out, out_idx


def get_file_list(path_prefix):
    file_list = []
    basepath = Path(path_prefix)
    files_in_basepath = basepath.iterdir()
    for item in files_in_basepath:
        if item.is_file() and ".xml" in item.name:
            file_list.append(f"{path_prefix}{item.name}")
        elif item.is_dir():
            file_list = file_list + get_file_list(f"{path_prefix}{item.name}\\")
    return file_list




xml_path = "xml_laws\\akn\\il\\act\\"
filenames = get_file_list(xml_path)
question_list = []
title_list = []
print("Start parse xml files...")
for law in filenames:
    with open(law, 'r', encoding='utf-8') as file:
        content = doc = file.read()
        original_law_content = BeautifulSoup(content, 'xml')
        title = original_law_content.find("title").find("p", recursive=True).text
        list_of_points = original_law_content.find("body").find("list", recursive=False)
        if not list_of_points:
            list_of_chapters = original_law_content.find("body").find_all("chapter", recursive=False)
            for chapter_id, chapter in enumerate(list_of_chapters):
                if chapter.find("list"):
                    points = chapter.find("list").find_all('point')
                    for point in points:
                        title_list.append(title)
                        question_list.append(point.text)
        else:
            points = list_of_points.find_all('point', recursive=False)
            for point in points:
                question_list.append(point.text)
                title_list.append(title)
print("Done parse xml files")

#Tokenizing with simple preprocess gensim's simple preprocess
def sent_to_words(sentences):
    for sentence in sentences:
        yield(simple_preprocess(str(sentence), deacc=True)) # returns lowercase tokens, ignoring tokens that are too short or too long

question_words = list(sent_to_words(question_list))

#Getting rid of stopwords
with open("heb_stopwords.txt", "r", encoding="utf-8") as file:
    stop_words = file.read().splitlines()


def remove_stopwords(sentence):
  filtered_words = [word for word in sentence if word not in stop_words]
  return filtered_words


print("Start remove stop words")
filtered_questions = [remove_stopwords(question) for question in question_words]
print("Done remove stop words")
del question_words
#Instantiating the model
n = 300

print("Start load W2V model")
model = get_word2vec_instance()
print("Done load W2V model")
word_vectors = model.wv


from sklearn.feature_extraction.text import TfidfVectorizer


merged_questions = [' '.join(question) for question in filtered_questions]
document_names = ['Doc {:d}'.format(i) for i in range(len(merged_questions))]


def get_tfidf(docs, ngram_range=(1,1), index=None):
    vect = TfidfVectorizer(ngram_range=ngram_range)
    return vect, vect.fit_transform(docs)


print("Start load TF IDF model")
vect, tfidf = get_tfidf(merged_questions, ngram_range=(1,1), index=document_names)
print("Done load TF IDF model")

def get_sent_embs(emb_model):
    def thread_function(vec_list, indexes):
        vect_feature_names = vect.get_feature_names()
        len_filt_q = len(filtered_questions)
        for desc in range(indexes[0], indexes[1]):
            zero_vec = np.zeros((1, n))
            if len(filtered_questions[desc]) > 0:
                # sent_emb = np.zeros((1, n))
                # div = 0
                model = emb_model
                if desc % 100 == 0:
                    print(f"get_sent_embs Doc: {desc} out of {len_filt_q}")
                itf_verctor = pd.DataFrame(tfidf[desc].todense(), columns=vect.get_feature_names()).T
                words = list(
                    filter(lambda x: x in model.wv.vocab and x in vect_feature_names, filtered_questions[desc]))
                words_emb = list(map(lambda x: model.wv[x], words))
                weights = list(map(lambda x: itf_verctor.loc[x][0], words))
                vecs = [((words_emb)[i] * weights[i])[:n] for i in range(len(words))]
                vecs.append(zero_vec)
                sent_emb = np.add.reduce(vecs)
                div = sum(weights)
            if div == 0:
                div += 1e-13
                print(desc)

            sent_emb = np.divide(sent_emb, div)
            vec_list.append(sent_emb.flatten())

    import threading
    num_of_threads = 8
    split_filtered_questions, indexes_list = chunkIt(filtered_questions, num=num_of_threads)
    thread_list = []
    sent_embs_list = [[] for i in range(num_of_threads)]
    for i in range(num_of_threads):
        thread_list.append(threading.Thread(target=thread_function, args=(sent_embs_list[i], indexes_list[i],)))
        thread_list[i].start()
    for thread in thread_list:
        thread.join()
    sent_embs = []
    for l in sent_embs_list:
        sent_embs = sent_embs + l
    return sent_embs


print("Start load vectors")
# ft_sent = get_sent_embs(emb_model=model)
ft_sent = list(np.load("testnp.npy"))
print("Done load vectors")

def get_n_most_similar(embeddings, n):
    """
    Takes the embedding vector of interest, the list with all embeddings, and the number of similar questions to
    retrieve.
    Outputs the disctionary IDs and distances
    """
    nbrs = NearestNeighbors(n_neighbors=n, metric='cosine').fit(embeddings)
    distances, indices = nbrs.kneighbors([embeddings[-1]])
    similar_indices = indices[0][1:]
    similar_distances = distances[0][1:]
    return similar_indices, similar_distances

def print_similar(embeddings, n):
    """
    Convenience function for visual analysis
    """
    closest_ind, closest_dist = get_n_most_similar(embeddings, n)
    # print('Question %s \n \n is most similar to these %s questions: \n' % (question_list[interest_index], n))
    for question in closest_ind:
        print('ID ', question, ': ',question_list[question], "from law:", title_list[question])


query = ""

while query != "quit":

    query = input("חיפוש בספר החוקים - ליציאה כתבו quit:\n")
    try:
        len_before_query = len(merged_questions)
        question_words = list(sent_to_words([query]))
        merged_questions2 = merged_questions + [query]
        document_names2 = document_names + [f"Doc {len_before_query}"]
        vect2, tfidf2 = get_tfidf(merged_questions2, ngram_range=(1,1), index=document_names2)
        zero_vec = np.zeros((1, n))
        desc = len(merged_questions)
        if len(query.split(" ")) > 0:
            itf_verctor = pd.DataFrame(tfidf2[desc].todense(), columns=vect2.get_feature_names()).T
            words = list(
                filter(lambda x: x in model.wv.vocab and x in vect2.get_feature_names(), query.split(" ")))
            words_emb = list(map(lambda x: model.wv[x], words))
            weights = list(map(lambda x: itf_verctor.loc[x][0], words))
            vecs = [((words_emb)[i] * weights[i])[:n] for i in range(len(words))]
            vecs.append(zero_vec)
            sent_emb = np.add.reduce(vecs)
            div = sum(weights)
        if div == 0:
            div += 1e-13
            print(desc)

        sent_emb = np.divide(sent_emb, div)
        ft_sent.append(sent_emb.flatten())
        print_similar(ft_sent, 5)
        ft_sent.pop()
    except Exception as e:
        print(f"Exceprion: {e}")

