from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch
from typing import List
import numpy as np


class Synonymizer():
    def __init__(self, prob, device):

        self.probability = prob 
        self.device = device

        # --- Auth & Model ------------------------------------------------------------
        LLAMA_TEXT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

        text_tokenizer = AutoTokenizer.from_pretrained(LLAMA_TEXT_MODEL, padding_side='left')

        if text_tokenizer.pad_token is None:
            text_tokenizer.pad_token = text_tokenizer.eos_token

        text_model = AutoModelForCausalLM.from_pretrained(
            LLAMA_TEXT_MODEL,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            cache_dir="/scratch/jocazar/jocazar"
        )
        text_model.eval()
        text_model.to(self.device)
        self.text_tokenizer = text_tokenizer
        self.text_model = text_model

    def decision(self):
        if self.probability is None:
            return False

        value = np.random.rand(1)[0]
        # print("value: ", value)
        # print("probability: ", self.probability)
        # print("value < self.probability: ", value < self.probability)
        return value < self.probability

    def synonymize_batch(self,
        sentences: List[str],
        max_changes: int = 3,
        temperature: float = 0.4,
        top_p: float = 0.9,
        max_new_tokens: int = 6,
        batch_size: int = 8,
    ) -> List[str]:
        """
        Batched version of synonymization.
        Replaces up to `max_changes` words in each sentence with close synonyms while preserving meaning.
        Returns a list of single-line, quoted outputs in the same order as `sentences`.
        """
        if not sentences:
            return []
        
        if self.decision() == False:
            # print("Not synonymizing this batch")
            return sentences

        sys_prompt = (
            "You are a careful paraphraser. Your task is to replace a FEW words "
            "with close synonyms while preserving exact meaning, tone, and grammar. "
            "Do NOT alter numbers, units, dates, or named entities. "
            f"Change at most {max_changes} words. Output ONLY the final sentence."
        )

        cleaned = [s.strip().strip('"').strip("'") for s in sentences]
        outputs: List[str] = []

        # Process in chunks to control memory usage
        for start in range(0, len(cleaned), batch_size):
            chunk = cleaned[start : start + batch_size]

            # Build per-item user prompts (needed for clean stripping later)
            user_prompts = [
                f"Rewrite by swapping a few words with close synonyms (at most {max_changes}).\n"
                f"Sentence: {text}"
                for text in chunk
            ]

            # Build chat messages per sample and render with chat template
            chats = []
            for up in user_prompts:
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": up},
                ]
                chat = self.text_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                chats.append(chat)

            # Tokenize as a batch (pad to max length)
            inputs = self.text_tokenizer(
                chats,
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                gen = self.text_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    eos_token_id=self.text_tokenizer.eos_token_id,
                    pad_token_id=self.text_tokenizer.eos_token_id,
                )

            # Decode each item, strip preamble using its own user prompt, tidy up
            decoded = self.text_tokenizer.batch_decode(gen, skip_special_tokens=True)
            for text_out, up in zip(decoded, user_prompts):
                out = text_out.split(up)[-1].strip()
                for tag in ["Assistant:", "assistant:", "Response:", "Output:", "assistant"]:
                    if out.startswith(tag):
                        out = out[len(tag):].strip()
                out = " ".join(out.split())
                outputs.append(out)

        return outputs
