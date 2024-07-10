import os
from typing import List, Dict, Tuple, Any, Optional

import cv2
import gradio as gr
import numpy as np
import som
import supervision as sv
import torch
from segment_anything import sam_model_registry

from sam_utils import sam_interactive_inference, sam_inference
from utils import postprocess_masks, Visualizer

HOME = "/content" 
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SAM_CHECKPOINT = os.path.join(HOME, "SoM/weights/sam_vit_h_4b8939.pth")
# SAM_CHECKPOINT = "weights/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"

ANNOTATED_IMAGE_KEY = "annotated_image"
DETECTIONS_KEY = "detections"
MARKDOWN = """
<div align='center'>
    <h1>
        <img 
            src='https://som-gpt4v.github.io/website/img/som_logo.png' 
            style='height:50px; display:inline-block'
        />  
        Set-of-Mark (SoM) Prompting Unleashes Extraordinary Visual Grounding in GPT-4V
    </h1>
    <br>
    [<a href="https://arxiv.org/abs/2109.07529"> arXiv paper </a>] 
    [<a href="https://som-gpt4v.github.io"> project page </a>]
    [<a href="https://github.com/roboflow/set-of-mark"> python package </a>]
    [<a href="https://github.com/microsoft/SoM"> code </a>]
</div>

## 🚧 Roadmap

- [ ] Support for alphabetic labels
- [ ] Support for Semantic-SAM (multi-level)
- [ ] Support for mask filtering based on granularity
"""

SAM = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT).to(device=DEVICE)


def inference(
    image_and_mask: Dict[str, np.ndarray],
    annotation_mode: List[str],
    mask_alpha: float
) -> Tuple[Tuple[np.ndarray, List[Tuple[np.ndarray, str]]], Dict[str, Any]]:
    image = image_and_mask['image']
    mask = cv2.cvtColor(image_and_mask['mask'], cv2.COLOR_RGB2GRAY)
    is_interactive = not np.all(mask == 0)
    visualizer = Visualizer(mask_opacity=mask_alpha)
    if is_interactive:
        detections = sam_interactive_inference(
            image=image,
            mask=mask,
            model=SAM)
    else:
        detections = sam_inference(
            image=image,
            model=SAM
        )
        detections = postprocess_masks(
            detections=detections)
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotated_image = visualizer.visualize(
        image=bgr_image,
        detections=detections,
        with_box="Box" in annotation_mode,
        with_mask="Mask" in annotation_mode,
        with_polygon="Polygon" in annotation_mode,
        with_label="Mark" in annotation_mode)
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    state = {
        ANNOTATED_IMAGE_KEY: annotated_image,
        DETECTIONS_KEY: detections
    }
    return (annotated_image, []), state


def prompt(
    message: str,
    history: List[List[str]],
    state: Dict[str, Any],
    api_key: Optional[str]
) -> str:
    if api_key == "":
        return "⚠️ Please set your OpenAI API key first"
    if state is None or ANNOTATED_IMAGE_KEY not in state:
        return "⚠️ Please generate SoM visual prompt first"
    return som.prompt_image(
        api_key=api_key,
        image=cv2.cvtColor(state[ANNOTATED_IMAGE_KEY], cv2.COLOR_BGR2RGB),
        prompt=message
    )


def on_image_input_clear():
    return None, {}


def highlight(
    state: Dict[str, Any],
    history: List[List[str]]
) -> Optional[Tuple[np.ndarray, List[Tuple[np.ndarray, str]]]]:
    if DETECTIONS_KEY not in state or ANNOTATED_IMAGE_KEY not in state:
        return None

    detections: sv.Detections = state[DETECTIONS_KEY]
    annotated_image: np.ndarray = state[ANNOTATED_IMAGE_KEY]

    if len(history) == 0:
        return None

    text = history[-1][-1]
    relevant_masks = som.extract_relevant_masks(
        text=text,
        detections=detections
    )
    relevant_masks = [
        (mask, mark)
        for mark, mask
        in relevant_masks.items()
    ]
    return annotated_image, relevant_masks


image_input = gr.Image(
    label="Input",
    type="numpy",
    tool="sketch",
    interactive=True,
    brush_radius=20.0,
    brush_color="#FFFFFF",
    height=512
)
checkbox_annotation_mode = gr.CheckboxGroup(
    choices=["Mark", "Polygon", "Mask", "Box"],
    value=['Mark'],
    label="Annotation Mode")
slider_mask_alpha = gr.Slider(
    minimum=0,
    maximum=1,
    value=0.05,
    label="Mask Alpha")
image_output = gr.AnnotatedImage(
    label="SoM Visual Prompt",
    color_map={
        str(i): sv.ColorPalette.default().by_idx(i).as_hex()
        for i in range(64)
    },
    height=512
)
openai_api_key = gr.Textbox(
    show_label=False,
    placeholder="Before you start chatting, set your OpenAI API key here",
    lines=1,
    type="password")
chatbot = gr.Chatbot(
    label="GPT-4V + SoM",
    height=256)
generate_button = gr.Button("Generate Marks")
highlight_button = gr.Button("Highlight Marks")

with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    inference_state = gr.State({})
    with gr.Row():
        with gr.Column():
            image_input.render()
            with gr.Accordion(
                    label="Detailed prompt settings (e.g., mark type)",
                    open=False):
                with gr.Row():
                    checkbox_annotation_mode.render()
                with gr.Row():
                    slider_mask_alpha.render()
        with gr.Column():
            image_output.render()
            generate_button.render()
            highlight_button.render()
    with gr.Row():
        openai_api_key.render()
    with gr.Row():
        gr.ChatInterface(
            chatbot=chatbot,
            fn=prompt,
            additional_inputs=[inference_state, openai_api_key])

    generate_button.click(
        fn=inference,
        inputs=[image_input, checkbox_annotation_mode, slider_mask_alpha],
        outputs=[image_output, inference_state])
    image_input.clear(
        fn=on_image_input_clear,
        outputs=[image_output, inference_state]
    )
    highlight_button.click(
        fn=highlight,
        inputs=[inference_state, chatbot],
        outputs=[image_output])

demo.queue().launch(debug=False, show_error=True, share=True)
