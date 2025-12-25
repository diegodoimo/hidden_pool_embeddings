import torch


with open("./results/my_inputs.txt", "r") as f:
    my_file = f.readlines()

with open("./results/st_inputs.txt", "r") as f:
    st_file = f.readlines()


my_q = torch.load("./results/my_queries.pt")
my_q_0 = torch.load("./results/my_q_0.pt", weights_only=False)
my_q_2 = torch.load("./results/my_q_1.pt", weights_only=False)
my_q_emb = torch.load("./results/my_q_embeddings.pt")
my_doc = torch.load("./results/my_docs.pt")

st_q = torch.load("./results/st_queries.pt")
st_q_0 = torch.load("./results/st_q_0.pt", weights_only=False)
st_q_2 = torch.load("./results/st_q_2.pt", weights_only=False)

st_q_emb = torch.load("./results/st_q_embeddings.pt")
st_q_emb_out = torch.load("./results/st_q_embeddings_out.pt")
st_doc = torch.load("./results/st_docs.pt")


my_0 = my_q_0.last_hidden_state
st_0 = st_q_0["token_embeddings"]
torch.testing.assert_close(my_0, st_0)

my_2 = my_q_2
st_2 = st_q_2["sentence_embedding"]
torch.testing.assert_close(my_2, st_2)


# *************************************


my_q = torch.load("./results/my_q_embeds.pt", weights_only=False)
my_d = torch.load("./results/my_doc_embeds.pt", weights_only=False)
my_q_idx_to_id = torch.load("./results/my_query_idx_to_id.pt", weights_only=False)
my_scores = torch.load("./results/my_scores.pt", weights_only=False)
my_res0 = torch.load("./results/my_results0.pt", weights_only=False)
my_res1 = torch.load("./results/my_results1.pt", weights_only=False)
my_qrels = torch.load("./results/my_qrels.pt", weights_only=False)

mteb_q = torch.load("./results/mteb_q_embeds.pt", weights_only=False)
mteb_d = torch.load("./results/mteb_doc_embeds.pt", weights_only=False)
mteb_q_idx_to_id = torch.load("./results/query_idx_to_id.pt", weights_only=False)
mteb_scores = torch.load("./results/mteb_scores.pt", weights_only=False)

mteb_res0 = torch.load("./results/mteb_results0.pt", weights_only=False)
mteb_res1 = torch.load("./results/mteb_results1.pt", weights_only=False)
mteb_qrels = torch.load("./results/mteb_qrels.pt", weights_only=False)


torch.testing.assert_close(my_q, torch.from_numpy(mteb_q))
torch.testing.assert_close(my_d, torch.from_numpy(mteb_d))
torch.testing.assert_close(my_scores, mteb_scores)

mteb_q_idx_to_id == my_q_idx_to_id
mteb_qrels == my_qrels


my_qrels


for key, val in my_res0.items():
    val_mteb = mteb_res1[key]
    assert False
val[key]

val_mteb[key]


max(val_mteb.values())
sorted_dict = dict(sorted(val_mteb.items(), key=lambda item: item[1], reverse=True))

list(sorted_dict.keys())[:20] == list(val.keys())[1:21]


val.keys()

my_scores
mteb_scores
