import pickle
from scipy.spatial.distance import cosine

from embeddings import get_embedding

cache_path = "example_dog_images/dog_embeddings.pkl"


def find_doggleganger(selfie_path, dog_embeddings):
    selfie_embedding = get_embedding(selfie_path)

    similarities = {}
    for dog_file, dog_embedding in dog_embeddings.items():
        similarity = 1 - cosine(selfie_embedding, dog_embedding)
        similarities[dog_file] = similarity

    best_match = max(similarities, key=similarities.get)
    return best_match, similarities[best_match]


# Usage
selfie_path = "path_to_selfie.jpg"

with open(cache_path) as f:
    dog_embeddings = pickle.load(f)

best_match, similarity = find_doggleganger(selfie_path, dog_embeddings)
print(f"Your doggleganger is {best_match} with similarity {similarity}")
