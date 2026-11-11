"""Gradio demo that shows each agent's output for one request."""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
from PIL import Image


def handle_request(hub, out_dir, trace_dir, image_path, text, use_critique):
    """Used by the button and by our tests. Always returns something the page can show."""
    state = RequestState(user_text=text or "", image_path=image_path, use_critique=bool(use_critique))
    state = run_request(state, hub, out_dir, trace_dir)
    planner_view = {"route": state.route, "reason": state.planner_reason, "image_quality": state.image_quality, "image_text_flagged": state.image_text_flagged}
    vision_view = state.vision.model_dump() if state.vision else {}
    prompt_view = {"final_prompt": state.prompt}
    generation_view = [g.model_dump() for g in state.generations]
    critique_view = [c.model_dump() for c in state.critiques]
    image = Image.open(state.final_image) if state.final_image else None
    trace_view = json.dumps([e.model_dump() for e in state.trace], indent=1, default=str)
    return f"{state.final_status}: {state.message}", image, planner_view, vision_view, prompt_view, generation_view, critique_view, trace_view


def build_app(hub, out_dir, trace_dir):
    out_dir, trace_dir = Path(out_dir), Path(trace_dir)
    with gr.Blocks(title="DATA 266 Lab 2 Part 3, Team 24") as demo:
        gr.Markdown("# Multimodal multi-agent demo\nUpload an image, type a request and see what each agent did.")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="filepath", label="Input image (optional for domain-style requests)")
                text = gr.Textbox(label="Request", placeholder="Turn this photo into a watercolor painting")
                critique = gr.Checkbox(value=True, label="Use the critique loop")
                button = gr.Button("Run", variant="primary")
            with gr.Column():
                status = gr.Textbox(label="Result")
                result = gr.Image(label="Final image")
        with gr.Row():
            planner_box = gr.JSON(label="1. Planner")
            vision_box = gr.JSON(label="2. Vision")
            prompt_box = gr.JSON(label="3. Prompt")
        with gr.Row():
            generation_box = gr.JSON(label="4. Generation")
            critique_box = gr.JSON(label="5. Critique")
        trace_box = gr.Code(label="Trace", language="json")

        def on_click(path, words, use):
            try:
                return handle_request(hub, out_dir, trace_dir, path, words, use)
            except Exception as err:  # the demo must fail with a message, not a stack trace
                return f"error: {type(err).__name__}: {err}", None, {}, {}, {}, [], [], ""

        button.click(on_click, [image, text, critique], [status, result, planner_box, vision_box, prompt_box, generation_box, critique_box, trace_box])
    return demo
