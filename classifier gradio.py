import gradio as gr
import fastbook
from fastai.vision.all import *
fastbook.setup_book()
from fastbook import *
import logging
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def classifier(input_img):
    current_dir = Path(__file__).resolve().parent
    learn_inf = load_learner(current_dir/'export.pkl')
    return learn_inf.predict(PILImage.create(input_img))

demo = gr.Interface(
    fn=classifier,
    inputs=gr.Image(shape=(224, 224)),
    outputs="text")
demo.launch()