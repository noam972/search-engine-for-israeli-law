from pathlib import Path

from bs4 import BeautifulSoup
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np


def split_list(seq, num):
    """
    Splitting a list into N parts of approximately equal length

    Args:
        seq(list): list to split
        num(int): Number of lists to split

    Returns(list, list):
        List of lists
        list of tuple that contain the indexes
    """
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    out_idx = []
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        out_idx.append((int(last), int(last + avg)))
        last += avg

    return out, out_idx


def get_file_list(dir_path):
    """
    Get law file path list
    Args:
        dir_path(str): directory path

    Returns(list):
        law file path list
    """
    file_list = []
    base_path = Path(dir_path)
    files_in_base_path = base_path.iterdir()
    for item in files_in_base_path:
        if item.is_file() and ".xml" in item.name:
            file_list.append(f"{dir_path}{item.name}")
        elif item.is_dir():
            file_list = file_list + get_file_list(f"{dir_path}{item.name}\\")
    return file_list


def get_points_list(file_path_list):
    """
        Get point list and law title list
    Args:
        file_path_list(list): List of laws xml path

    Returns:
        points_list(list): list of all points
        title_list(list): list of all law titles
    """
    points_list = []
    title_list = []
    for law in file_path_list:
        with open(law, 'r', encoding='utf-8') as file:
            content = file.read()
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
                            points_list.append(point.text)
            else:
                points = list_of_points.find_all('point', recursive=False)
                for point in points:
                    points_list.append(point.text)
                    title_list.append(title)
    return points_list, title_list


# Tokenizing with simple preprocess gensim's simple preprocess
def sent_to_words(sentences):
    for sentence in sentences:
        # returns lowercase tokens, ignoring tokens that are too short or too long
        yield simple_preprocess(str(sentence), deacc=True)


# Getting rid of stopwords
def remove_stopwords(points_words):
    """
    Remove stop words from list
    Args:
        points_words(list): List of list to remove stop words from.

    Returns:
        list of lists without stopwords
    """
    with open("heb_stopwords.txt", "r", encoding="utf-8") as file:
        stop_words = file.read().splitlines()

    def remove_stopwords_from_sentence(sentence):
        filtered_words = [word for word in sentence if word not in stop_words]
        return filtered_words

    return [remove_stopwords_from_sentence(question) for question in points_words]


def get_tfidf(docs):
    """
    Get tfidf instance and metric
    Args:
        docs(list): list of strings with points content

    Returns:
        vector(TfidfVectorizer): TfidfVectorizer instance
        matrix: tfidf sparse matrix
    """
    vector = TfidfVectorizer(ngram_range=(1, 1))
    return vector, vector.fit_transform(docs)


def get_sent_embs(emb_model, vect, tfidf, filtered_points, n, num_of_threads=8):
    """

    Args:
        emb_model(): Word2vec model
        vect(TfidfVectorizer): TfidfVectorizer instance
        tfidf(): tfidf sparse matrix
        filtered_points(list): list of lists without stopwords
        n(int): Vectors size
        num_of_threads(int): Number of thread to run

    Returns:

    """
    def thread_function(vec_list, indexes):
        vect_feature_names = vect.get_feature_names()
        len_filt_q = len(filtered_points)
        sent_emb = None
        for desc in range(indexes[0], indexes[1]):
            div = 0
            zero_vec = np.zeros((1, n))
            if len(filtered_points[desc]) > 0:
                # sent_emb = np.zeros((1, n))
                # div = 0
                model = emb_model
                if desc % 100 == 0:
                    print(f"get_sent_embs Doc: {desc} out of {len_filt_q}")
                itf_verctor = pd.DataFrame(tfidf[desc].todense(), columns=vect.get_feature_names()).T
                words = list(
                    filter(lambda x: x in model.vocab and x in vect_feature_names, filtered_points[desc]))
                words_emb = list(map(lambda x: model[x], words))
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
    split_filtered_questions, indexes_list = split_list(filtered_points, num=num_of_threads)
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
