from pre_processing_functions import get_file_list, get_points_list, sent_to_words, remove_stopwords, get_tfidf, get_sent_embs
from word2vec import get_word2vec_instance
import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors


def get_n_most_similar(embeddings, n):
    """
    Takes the embedding vector of interest, the list with all embeddings, and the number of similar questions to
    retrieve.
    Outputs the dictionary IDs and distances
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
    for question in closest_ind:
        print("===============================================")
        print("שם החוק:")
        print(title_list[question])
        print("====")
        print("סעיף:")
        print(points_list[question])
        print("===============================================")


xml_path = "xml_laws\\akn\\il\\act\\"
n = 300

print("Start parse xml files...")
file_path_list = get_file_list(xml_path)
points_list, title_list = get_points_list(file_path_list)
print("Done parse xml files")


points_words = list(sent_to_words(points_list))

print("Start remove stop words")
filtered_points = remove_stopwords(points_words)
print("Done remove stop words")
del points_words

print("Start load W2V model")
model = get_word2vec_instance()
print("Done load W2V model")

# lists of points content
merged_points = [' '.join(question) for question in filtered_points]
document_names = ['Doc {:d}'.format(i) for i in range(len(merged_points))]

print("Start load TF IDF model")
vect, tfidf = get_tfidf(merged_points)
print("Done load TF IDF model")

print("Start load vectors")
# ft_sent = get_sent_embs(emb_model=model, vect=vect, tfidf=tfidf, filtered_points=filtered_points, n=n)
ft_sent = list(np.load("vectors200.npy"))
print("Done load vectors")

query = ""

while True:

    query = input("חיפוש בספר החוקים - ליציאה כתבו quit:\n")
    if "quit" in query:
        break
    try:
        sent_emb = None
        div = 0
        len_before_query = len(merged_points)
        points_words = list(sent_to_words([query]))
        merged_questions2 = merged_points + [query]
        document_names2 = document_names + [f"Doc {len_before_query}"]
        vect2, tfidf2 = get_tfidf(merged_questions2)
        zero_vec = np.zeros((1, n))
        desc = len(merged_points)
        if len(query.split(" ")) > 0:
            itf_verctor = pd.DataFrame(tfidf2[desc].todense(), columns=vect2.get_feature_names()).T
            words = list(
                filter(lambda x: x in model.vocab and x in vect2.get_feature_names(), query.split(" ")))
            words_emb = list(map(lambda x: model[x], words))
            weights = list(map(lambda x: itf_verctor.loc[x][0], words))
            vecs = [((words_emb)[i] * weights[i])[:n] for i in range(len(words))]
            vecs.append(zero_vec)
            sent_emb = np.add.reduce(vecs)
            div = sum(weights)
        if div == 0:
            div += 1e-13
            print("אחת המילים אינה חוקית.")
            continue

        sent_emb = np.divide(sent_emb, div)
        ft_sent.append(sent_emb.flatten())
        print_similar(ft_sent, 5)
        ft_sent.pop()
    except Exception as e:
        print(f"Exceprion: {e}")

