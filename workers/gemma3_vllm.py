import torch
import torch.nn as nn
from transformers import AutoTokenizer
from workers.base import BaseWorker
from vllm import LLM, SamplingParams


class VLLMAdapter(nn.Module):
    def __init__(
        self,
        model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
    ):
        super().__init__()
        print(f"[VLLMAdapter] Initializing vLLM from: {model_path}")

        self.llm = LLM(
            model=model_path,
            tokenizer=model_path,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="bfloat16",
            enforce_eager=False,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.config = type("Config", (), {"architectures": ["VLLM"]})()
        self.dummy_param = nn.Parameter(torch.tensor([0.0]))

    def forward(self):
        return None

    def generate(self, input_ids, **kwargs):
        max_new_tokens = kwargs.get("max_new_tokens", 2048)

        stop_token_ids = [self.tokenizer.eos_token_id]
        if "<end_of_turn>" in self.tokenizer.get_vocab():
            stop_token_ids.append(self.tokenizer.convert_tokens_to_ids("<end_of_turn>"))

        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            ignore_eos=False,
        )

        prompt_token_ids_list = input_ids.cpu().tolist()

        vllm_inputs = [{"prompt_token_ids": ids} for ids in prompt_token_ids_list]

        outputs = self.llm.generate(
            prompts=vllm_inputs, sampling_params=sampling_params, use_tqdm=False
        )

        result_sequences = []
        max_len = 0
        for i, output in enumerate(outputs):
            generated_ids = list(output.outputs[0].token_ids)
            full_ids = prompt_token_ids_list[i] + generated_ids
            result_sequences.append(full_ids)
            max_len = max(max_len, len(full_ids))

        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )
        padded_results = []
        for seq in result_sequences:
            pad_len = max_len - len(seq)
            padded_seq = seq + [pad_token_id] * pad_len
            padded_results.append(padded_seq)

        return torch.tensor(padded_results, device="cuda")


class Gemma3VLLMWorker(BaseWorker):
    def load_model_and_tokenizer(self, load_config):
        model_path = load_config["config_dir"]

        model = VLLMAdapter(model_path, gpu_memory_utilization=0.9)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side="left"
        )
        if hasattr(tokenizer, "tokenizer"):
            tokenizer = tokenizer.tokenizer

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    @property
    def system_prompt(self):
        return "你是一个专业的医学智能助手。请根据医学知识准确回答问题。\n"

    @property
    def instruction_template(self):
        return (
            "<start_of_turn>user\n"
            + self.system_prompt
            + "{instruction}<end_of_turn>\n"
            + "<start_of_turn>model\n"
        )

    @property
    def instruction_template_with_fewshot(self):
        return (
            "<start_of_turn>user\n"
            + self.system_prompt
            + "{fewshot_examples}"
            + "{instruction}<end_of_turn>\n"
            + "<start_of_turn>model\n"
        )

    @property
    def fewshot_template(self):
        return (
            "问题：{user}<end_of_turn>\n"
            + "<start_of_turn>model\n"
            + "答案：{gpt}<end_of_turn>\n"
            + "<start_of_turn>user\n"
        )

# accelerate launch \
    # --num_processes=1 \
    # ./src/generate_answers.py \
    # --use_cot \
    # --model_id="gemma3_vllm" \
    # --input_path='./data/CMB/CMB-Exam/CMB-test/CMB-test-choice-question-merge.json' \
    # --output_path='./result/Exam/gemma3_vllm/modelans.json' \
    # --batch_size 64 \
    # --model_config_path="./configs/model_config.yaml"