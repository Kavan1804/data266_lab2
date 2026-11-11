"""Real model backends for Part 3. One Stable Diffusion pipeline is loaded at a time."""
from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from PIL import Image

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
PIX2PIX_MODEL = "timbrooks/instruct-pix2pix"
VLM_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
CLIP_MODEL = "openai/clip-vit-base-patch32"


# Used when eval/p2_final_model.json does not exist yet: Kavan's own Part 2 baseline LoRA.
DEFAULT_LORA = {"run_name": "kavan_baseline", "hf_repo": "Kavan1804/data266-p2-baseline-kavan-lora",
                "local_path": "outputs/p2_baseline_kavan/lora", "weight_name": "pytorch_lora_weights.safetensors",
                "scale": 1.0, "prompt_hint": "", "is_final": False}


def find_lora(roots: list[Path], config: dict | None) -> dict | None:
    """Where to load the Part 2 LoRA from. See TEAM_WORKPLAN.md Section 6.4.

    config is eval/p2_final_model.json (built into the notebook), or DEFAULT_LORA before the merge.
    A local_path that exists under one of the roots wins (same machine as the Part 2 run).
    Otherwise the LoRA is downloaded from hf_repo on Hugging Face.
    """
    config = config or DEFAULT_LORA
    weight = config.get("weight_name", "pytorch_lora_weights.safetensors")
    source, where = None, ""
    if config.get("local_path"):
        for root in roots:
            folder = Path(root) / config["local_path"]
            if (folder / weight).exists():
                source, where = str(folder), "local file"
                break
    if source is None and config.get("hf_repo"):
        source, where = config["hf_repo"], "Hugging Face"
    if source is None:
        return None
    return {"source": source, "where": where, "weight_name": weight, "scale": float(config.get("scale", 1.0)),
            "prompt_hint": config.get("prompt_hint", ""), "run_name": config.get("run_name", ""), "is_final": bool(config.get("is_final", False))}


class ModelHub:
    def __init__(self, lora: dict | None, steps: int = 30, device: str | None = None):
        from transformers import AutoProcessor, CLIPModel, CLIPProcessor, Qwen2_5_VLForConditionalGeneration

        self.steps = steps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(VLM_MODEL, torch_dtype=self.dtype).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(VLM_MODEL, min_pixels=256 * 256, max_pixels=1024 * 1024)
        self.clip = CLIPModel.from_pretrained(CLIP_MODEL).to(self.device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        self.pipe, self.pipe_name = None, ""
        self.lora = lora

    # ---- vision language model
    def _chat(self, content: list[dict], max_new_tokens: int) -> str:
        from qwen_vl_utils import process_vision_info

        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.vlm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    def text(self, prompt: str, max_new_tokens: int = 128) -> str:
        return self._chat([{"type": "text", "text": prompt}], max_new_tokens)

    def vision(self, prompt: str, image_path: str, max_new_tokens: int = 128) -> str:
        image = Image.open(image_path).convert("RGB")
        return self._chat([{"type": "image", "image": image}, {"type": "text", "text": prompt}], max_new_tokens)

    def clip_score(self, image_path: str, text: str) -> float:
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(text=[text], images=image, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.inference_mode():
            out = self.clip(**inputs)
        a = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        b = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return float((a * b).sum())

    # ---- diffusion, one pipeline in memory at a time
    def _load(self, name: str):
        if self.pipe_name == name:
            return self.pipe
        self.pipe = None
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInstructPix2PixPipeline, StableDiffusionPipeline

        if name == "pix2pix":
            pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(PIX2PIX_MODEL, torch_dtype=self.dtype)
        elif name in ("img2img", "domain_img2img"):
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(BASE_MODEL, torch_dtype=self.dtype)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=self.dtype)
        if name.startswith("domain"):
            if self.lora is None:
                raise FileNotFoundError("No Part 2 LoRA found. Run p2_baseline_kavan.ipynb (it uploads one) or set P2_LORA in this notebook.")
            pipe.load_lora_weights(self.lora["source"], weight_name=self.lora["weight_name"])
        pipe.set_progress_bar_config(disable=True)
        self.pipe, self.pipe_name = pipe.to(self.device), name
        return self.pipe

    def generate(self, mode: str, prompt: str, image_path: str | None, seed: int, user_text: str = "", round_no: int = 0):
        generator = torch.Generator(self.device).manual_seed(seed)
        meta = {"steps": self.steps}
        if mode == "img2img":
            pipe = self._load("img2img")
            init = Image.open(image_path).convert("RGB").resize((512, 512))
            strength = min(0.6 + 0.1 * round_no, 0.8)
            out = pipe(prompt=prompt, image=init, strength=strength, guidance_scale=7.5, num_inference_steps=self.steps, generator=generator)
            meta.update(model=BASE_MODEL, strength=strength)
        elif mode == "edit":
            pipe = self._load("pix2pix")
            init = Image.open(image_path).convert("RGB").resize((512, 512))
            # instruct-pix2pix wants the instruction itself, so the user's words go first
            out = pipe(prompt=f"{user_text}. {prompt}" if round_no else user_text, image=init, image_guidance_scale=1.5 - 0.1 * round_no,
                       guidance_scale=7.5 + round_no, num_inference_steps=self.steps, generator=generator)
            meta.update(model=PIX2PIX_MODEL)
        elif mode == "domain_img2img":
            pipe = self._load("domain_img2img")
            init = Image.open(image_path).convert("RGB").resize((512, 512))
            strength = min(0.7 + 0.1 * round_no, 0.85)
            out = pipe(prompt=self._hint(prompt), image=init, strength=strength, guidance_scale=7.5, num_inference_steps=self.steps,
                       generator=generator, cross_attention_kwargs={"scale": self.lora["scale"]})
            meta.update(model=f"{BASE_MODEL} + {self.lora['source']}", strength=strength)
        else:
            pipe = self._load("domain")
            out = pipe(prompt=self._hint(prompt), guidance_scale=7.5, num_inference_steps=self.steps, generator=generator, height=512, width=512,
                       cross_attention_kwargs={"scale": self.lora["scale"]})
            meta.update(model=f"{BASE_MODEL} + {self.lora['source']}")
        flagged = out.nsfw_content_detected
        meta["nsfw"] = bool(flagged and flagged[0])
        return out.images[0], meta

    def _hint(self, prompt: str) -> str:
        hint = (self.lora or {}).get("prompt_hint", "")
        return f"{hint} {prompt}".strip() if hint and hint != "..." else prompt
