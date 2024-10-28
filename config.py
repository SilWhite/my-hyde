# use DeepSeek LLM
# KEY = 'sk-b0d1a733e4a44c4c91aaca082e3f9ed0' # my key, free tokens are unavailble
key = "sk-77ca6e57c7324f31a7b5ef9fc6e9b29d"  # help from friend, 2024-11-23 16:20:16 unavailabled
model = "deepseek-chat"  # only one model of deepseek is supported for now


api_mode = "prefix"  # chat or fim or prefix
task_lable = "web search"  # web search, scifact, arguana, trec-covid, fiqa, dbpedia-entity, trec-news, mr-tydi


# facebook-contriever
encoder_dir = "./facebook-contriever"
# contriever_msmarco_index
search_index_dir = "./contriever_msmarco_index"
# msmarco-v1-passage
corpus_dir = "./lucene-inverted.msmarco-v1-passage.20221004.252b5e"


# example system prompt: "You are a helpful assistant."
# each message have 3 roles: system, user, assistant
def get_prefix_msg(prompt):
    prefix_msg = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        # {
        #     "role": "user",
        #     "content": prompt,
        # },
        {
            "role": "assistant",
            "content": prompt,
            "prefix": True,  # must be True for prefix message
        },
    ]
    return prefix_msg


def get_chat_msg(prompt):
    chat_msg = [
        # {
        #     "role": "system",
        #     "content": "",
        # },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    return chat_msg
