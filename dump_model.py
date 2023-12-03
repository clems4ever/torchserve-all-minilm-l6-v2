import sentence_transformers 
from sentence_transformers import SentenceTransformer, CrossEncoder

# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'Each sentence is converted']

# Load model from HuggingFace Hub
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Create embeddigs + normalisation
model.encode(sentences=sentences, normalize_embeddings=True, convert_to_tensor= True)

print("Sentence embeddings has been computed successfully, the model is working!")

# We can now dump the model on disk
model.save("./embedder")

question = "how are you doing?"
model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")
inputs = [[question, sentence] for sentence in sentences]
out = model.predict(inputs)
print(out)
model.save("./cross_encoder")