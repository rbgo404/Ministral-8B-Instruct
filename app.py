from vllm import LLM
from vllm.sampling_params import SamplingParams
from transformers import AutoTokenizer

class InferlessPythonModel:
    def initialize(self):
        model_id = "mistralai/Ministral-8B-Instruct-2410"
        self.llm = LLM(model=model_id,gpu_memory_utilization=0.9,max_model_len=5000)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def infer(self, inputs):
        start_time = time.perf_counter()
        prompts = inputs["prompt"]
        temperature = inputs.get("temperature",0.7)
        top_p = inputs.get("top_p",0.1)
        repetition_penalty = inputs.get("repetition_penalty",1.18)
        top_k = inputs.get("top_k",40)
        max_tokens = inputs.get("max_tokens",256)

        sampling_params = SamplingParams(temperature=temperature,top_p=top_p,repetition_penalty=repetition_penalty,
                                         top_k=top_k,max_tokens=max_tokens)
        input_text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompts}], tokenize=False)
#         result = self.llm.generate(input_text, sampling_params)
#         result_output = [output.outputs[0].text for output in result]

        result = self.llm.generate(input_text, sampling_params)
        request_time = time.perf_counter() - start_time
        result_output = [[output.outputs[0].text,output.outputs[0].token_ids] for output in result]

        request_time = time.perf_counter() - start_time

        return {'tok_count': len(result_output[0][1]),
            'time': request_time,
#             'question': prompt,
            'answer': result_output[0][0],
            
        }

#         return {'result': result}

    def finalize(self):
        self.llm = None
