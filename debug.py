from transformers import AutoModel


encoder = AutoModel.from_pretrained("google/t5gemma-2-270m-270m")



encoder.decoder.embed_tokens.weights