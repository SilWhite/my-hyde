import time
import openai

# import cohere


class Generator:

    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.api_key = api_key

    def generate(self):
        return ""


# use deepseek llm
class DeepSeekGenerator(Generator):

    def __init__(
        self,
        model_name,  # deepseek-chat
        api_key,
        frequency_penalty=0.0,
        max_tokens=512,  # 是否需要限制？
        presence_penalty=0.0,
        # response_format={"type": "text"},  # deepseek默认为text
        stop=["\n\n\n"],
        stream=False,
        temperature=1.5,  # 通用对话: 1.3, 创意类写作/诗歌创作: 1.5
        top_p=1,  # 不建议同时对temperature和top_p进行修改
        wait_till_success=False,  # 是否需要等待成功
        api_mode="chat",  # deepseek api模式，chat或fim或prefix
    ):
        super().__init__(model_name, api_key)
        self.frequency_penalty = frequency_penalty
        self.max_tokens = max_tokens
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.stream = stream
        self.temperature = temperature
        self.top_p = top_p
        self.wait_till_success = wait_till_success
        self.api_mode = api_mode

    def parse_response(self, response):
        if self.api_mode == "chat" or self.api_mode == "prefix":
            return [response.choices[0].message.content]
        elif self.api_mode == "fim":
            return [response.choices[0].text]

    def generate(self, prompt):
        get_results = False

        if self.api_mode == "chat":
            base_url = "https://api.deepseek.com"
            message = [
                # {
                #     "role": "system",
                #     "content": "",
                # },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        elif self.api_mode == "fim":
            base_url = "https://api.deepseek.com/beta"
            message = prompt
        elif self.api_mode == "prefix":
            base_url = "https://api.deepseek.com/beta"
            message = [
                # {
                #     "role": "system",
                #     "content": "",
                # },
                # {
                #     "role": "user",
                #     "content": prompt,
                # },
                {
                    "role": "assistant",
                    "content": prompt,
                    "prefix": True,  # 必须为True，表示是前缀模式
                },
            ]
        else:
            raise ValueError("Invalid api_mode: {}".format(self.api_mode))

        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=base_url,
        )

        while not get_results:
            try:
                # 使用chat模式（FIM模式目前beta，可以测试对比）
                if (
                    self.api_mode == "chat" or self.api_mode == "prefix"
                ):  # 前缀模式沿用chat api
                    result = client.chat.completions.create(
                        model=self.model_name,
                        messages=message,
                        frequency_penalty=self.frequency_penalty,
                        max_tokens=self.max_tokens,
                        presence_penalty=self.presence_penalty,
                        stop=self.stop,
                        stream=self.stream,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        # logprobs=True,  # 感觉没什么用
                    )
                elif self.api_mode == "fim":
                    result = client.completions.create(
                        model="deepseek-chat",
                        prompt=message,
                        echo=False,
                        frequency_penalty=self.frequency_penalty,
                        # logit_bias # 可以返回多个，后续添加
                        max_tokens=self.max_tokens,
                        presence_penalty=self.presence_penalty,
                        stop=self.stop,
                        stream=self.stream,
                        # suffix= # 补全的后缀，本次任务不需要
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                elif self.api_mode == "prefix":
                    pass
                else:
                    raise ValueError("Invalid api_mode: {}".format(self.api_mode))

                get_results = True
            except Exception as e:
                if self.wait_till_success:
                    time.sleep(1)
                else:
                    raise e
        return self.parse_response(result)


# class OpenAIGenerator(Generator):
#     def __init__(self, model_name, api_key, n=8, max_tokens=512, temperature=0.7, top_p=1, frequency_penalty=0.0, presence_penalty=0.0, stop=['\n\n\n'], wait_till_success=False):
#         super().__init__(model_name, api_key)
#         self.n = n
#         self.max_tokens = max_tokens
#         self.temperature = temperature
#         self.top_p = top_p
#         self.frequency_penalty = frequency_penalty
#         self.presence_penalty = presence_penalty
#         self.stop = stop
#         self.wait_till_success = wait_till_success

#     @staticmethod
#     def parse_response(response):
#         to_return = []
#         for _, g in enumerate(response['choices']):
#             text = g['text']
#             logprob = sum(g['logprobs']['token_logprobs'])
#             to_return.append((text, logprob))
#         texts = [r[0] for r in sorted(to_return, key=lambda tup: tup[1], reverse=True)]
#         return texts

#     def generate(self, prompt):
#         get_results = False
#         while not get_results:
#             try:
#                 result = openai.Completion.create(
#                     engine=self.model_name,
#                     prompt=prompt,
#                     api_key=self.api_key,
#                     max_tokens=self.max_tokens,
#                     temperature=self.temperature,
#                     frequency_penalty=self.frequency_penalty,
#                     presence_penalty=self.presence_penalty,
#                     top_p=self.top_p,
#                     n=self.n,
#                     stop=self.stop,
#                     logprobs=1
#                 )
#                 get_results = True
#             except Exception as e:
#                 if self.wait_till_success:
#                     time.sleep(1)
#                 else:
#                     raise e
#         return self.parse_response(result)

# class CohereGenerator(Generator):
#     def __init__(self, model_name, api_key, n=8, max_tokens=512, temperature=0.7, p=1, frequency_penalty=0.0, presence_penalty=0.0, stop=['\n\n\n'], wait_till_success=False):
#         super().__init__(model_name, api_key)
#         self.cohere = cohere.Cohere(self.api_key)
#         self.n = n
#         self.max_tokens = max_tokens
#         self.temperature = temperature
#         self.p = p
#         self.frequency_penalty = frequency_penalty
#         self.presence_penalty = presence_penalty
#         self.stop = stop
#         self.wait_till_success = wait_till_success

#     @staticmethod
#     def parse_response(response):
#         text = response.generations[0].text
#         return text

#     def generate(self, prompt):
#         texts = []
#         for _ in range(self.n):
#             get_result = False
#             while not get_result:
#                 try:
#                     result = self.cohere.generate(
#                         prompt=prompt,
#                         model=self.model_name,
#                         max_tokens=self.max_tokens,
#                         temperature=self.temperature,
#                         frequency_penalty=self.frequency_penalty,
#                         presence_penalty=self.presence_penalty,
#                         p=self.p,
#                         k=0,
#                         stop=self.stop,
#                     )
#                     get_result = True
#                 except Exception as e:
#                     if self.wait_till_success:
#                         time.sleep(1)
#                     else:
#                         raise e
#             text = self.parse_response(result)
#             texts.append(text)
#         return texts
