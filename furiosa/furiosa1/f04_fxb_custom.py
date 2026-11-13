from furiosa_llm import LLM, SamplingParams

llm = LLM("Qwen/Qwen3-8B-FP8", fxb="/root/qwen3-8b-fp8.fxb")

params = SamplingParams(max_tokens=1000, top_p=0.3, top_k=100)

message = [{"role":"user", "content":"오늘 엔비디아 주가가 어떻게 돼?"}]
prompt = llm.tokenizer.apply_chat_template(message, tokenize=False)


response = llm.generate([prompt], params)

print(response[0].outputs[0].text)