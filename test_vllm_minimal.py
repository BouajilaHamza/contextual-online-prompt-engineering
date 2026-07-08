import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("vllm==0.5.4")
app = modal.App("test-vllm")

@app.function(image=image, gpu="A10G")
def test():
    from vllm import LLM, SamplingParams
    print("Loading model...")
    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True, gpu_memory_utilization=0.5)
    print("Model loaded.")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
    outputs = llm.generate("Hello, how are you?", sampling_params)
    for output in outputs:
        print(output.outputs[0].text)

if __name__ == "__main__":
    with app.run():
        test.remote()
