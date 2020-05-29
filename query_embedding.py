import time
from flask import Flask, request, jsonify
from one_sentence_embedding import one_sentence_embedding
from bm25_weighting import bm25_weighting
from preprocessing_utils import preprocessing_utils

# http://127.0.0.1:5000
app = Flask(__name__)

# These are some metadata of our word embedding
model_file = "cord19-300d.bin" # The file that we'll load the word embedding
dimension = 300 # The dimension of the word embedding

# load the word2vec model
start = time.time()
print("Start to load bm25 weightings of our corpus")
bm25wei = bm25_weighting("idf_score_para", 1.2, 0.75, preprocessing_utils(), None)
print("Start to load a " + str(dimension) + " dimension word embedding model from file " + model_file)
ose = one_sentence_embedding(bm25wei, model_file, dimension)
end = time.time()
print("It takes " + str(end - start) + " seconds to load the bm25 weighting and the model")


@app.route("/")
def query():
    return 'query_page.html'

# This takes a pair of param whose key is "query" and value is "<any query you want>"
# It will return a dictionary whose key is "vector" and value is a list of floats that stands for
# the embedding of the query
@app.route("/results")
def results():
    global ose
    """Generate a result set for a query and present the 10 results starting with <page_num>."""
    res = {}
    res["vector"] = ose.sentence_to_vector(request.args.get("query"))
    return res

# This is the main funtion
if __name__ == "__main__":
    app.run()
