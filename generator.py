import time
import openai

import config


# use deepseek LLM(https://www.deepseek.com/)
class DeepSeekGenerator:
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
        if api_mode not in ["chat", "fim", "prefix"]:
            raise ValueError("Invalid api_mode: {}".format(api_mode))

        self.model_name = model_name
        self.api_key = api_key
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
        else:
            raise ValueError(
                "Invalid api_mode: {}".format(self.api_mode)
            )  # won't run here

    def generate(self, prompt):
        get_results = False

        # chat mode 可以选填system prompt
        # fim mode 只有prompt字段
        # prefix mode 必须有assistant prompt，同时必须设置prefix=True
        if self.api_mode == "chat":
            base_url = "https://api.deepseek.com"
            message = config.get_chat_msg(prompt)
        elif self.api_mode == "fim":
            base_url = "https://api.deepseek.com/beta"
            message = prompt
        elif self.api_mode == "prefix":
            base_url = "https://api.deepseek.com/beta"
            message = config.get_prefix_msg(prompt)
        else:
            raise ValueError(
                "Invalid api_mode: {}".format(self.api_mode)
            )  # won't run here

        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=base_url,
        )

        while not get_results:
            try:
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
                        max_tokens=self.max_tokens,
                        presence_penalty=self.presence_penalty,
                        stop=self.stop,
                        stream=self.stream,
                        # suffix= # 补全的后缀，本次任务不需要
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                else:
                    raise ValueError("Invalid api_mode: {}".format(self.api_mode))

                get_results = True
            except Exception as e:
                if self.wait_till_success:
                    print("Error: {}".format(e))
                    print("Waiting for 1 second before retrying...")
                    time.sleep(1)
                else:
                    raise e
        return self.parse_response(result)
