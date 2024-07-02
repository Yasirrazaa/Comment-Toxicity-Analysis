import gradio as gr
from comment_toxicity.pipeline.prediction import PredictionPipeline
obj=PredictionPipeline()
interface=gr.Interface(fn=obj.model_prediction,inputs=gr.Textbox(lines=2,placeholder='comment to score'),outputs='text')
interface.launch(share=True)