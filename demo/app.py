#!/usr/bin/env python3
"""Gradio demo for multi-modal scene understanding."""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import get_device, suppress_warnings
from models import SceneUnderstandingModel
from viz import visualize_attention, visualize_object_detections


class SceneUnderstandingDemo:
    """Demo class for scene understanding model."""
    
    def __init__(self, config_path: str = "configs/model/default.yaml"):
        """Initialize demo.
        
        Args:
            config_path: Path to model configuration.
        """
        self.config = OmegaConf.load(config_path)
        self.device = get_device()
        
        # Suppress warnings
        suppress_warnings()
        
        # Load model
        self.model = self._load_model()
        
        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.text_model)
        
        # Scene and object class names
        self.scene_classes = ["indoor", "outdoor"]
        self.object_classes = [
            "person", "chair", "table", "sofa", "bed", "lamp", "tv", "book", "cat", "dog"
        ]
    
    def _load_model(self) -> SceneUnderstandingModel:
        """Load the trained model.
        
        Returns:
            Loaded model.
        """
        model = SceneUnderstandingModel(
            vision_model=self.config.model.vision_model,
            text_model=self.config.model.text_model,
            num_scene_classes=self.config.model.num_scene_classes,
            num_object_classes=self.config.model.num_object_classes,
            hidden_size=self.config.model.hidden_size,
            dropout=self.config.model.dropout
        ).to(self.device)
        
        # Try to load checkpoint
        checkpoint_path = "checkpoints/best_model.pt"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded model from {checkpoint_path}")
        else:
            print("No checkpoint found, using random weights")
        
        model.eval()
        return model
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input.
        
        Args:
            image: Input PIL image.
            
        Returns:
            Preprocessed image tensor.
        """
        # Convert to RGB
        image = image.convert("RGB")
        
        # Resize
        image = image.resize((224, 224))
        
        # Convert to numpy and normalize
        image_np = np.array(image).astype(np.float32) / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = (image_np - mean) / std
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def preprocess_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess text for model input.
        
        Args:
            text: Input text string.
            
        Returns:
            Tuple of (input_ids, attention_mask).
        """
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return (
            encoding["input_ids"].to(self.device),
            encoding["attention_mask"].to(self.device)
        )
    
    def predict_scene(
        self, 
        image: Image.Image, 
        text: str,
        show_attention: bool = True
    ) -> Tuple[str, str, str, Optional[np.ndarray]]:
        """Predict scene understanding from image and text.
        
        Args:
            image: Input image.
            text: Input text description.
            show_attention: Whether to show attention visualization.
            
        Returns:
            Tuple of (scene_prediction, object_predictions, description, attention_image).
        """
        # Preprocess inputs
        image_tensor = self.preprocess_image(image)
        input_ids, attention_mask = self.preprocess_text(text)
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(image_tensor, input_ids, attention_mask)
            
            # Get predictions
            scene_logits = outputs["scene_logits"]
            object_logits = outputs["object_logits"]
            
            # Scene prediction
            scene_pred = torch.argmax(scene_logits, dim=1).item()
            scene_confidence = torch.softmax(scene_logits, dim=1)[0, scene_pred].item()
            scene_prediction = f"{self.scene_classes[scene_pred]} (confidence: {scene_confidence:.3f})"
            
            # Object predictions
            object_probs = torch.sigmoid(object_logits)[0]
            object_predictions = []
            for i, prob in enumerate(object_probs):
                if prob > 0.5:  # Threshold for object presence
                    object_predictions.append(f"{self.object_classes[i]}: {prob:.3f}")
            
            object_text = ", ".join(object_predictions) if object_predictions else "No objects detected"
            
            # Generate description (simplified)
            description = f"The scene appears to be {self.scene_classes[scene_pred]} with the following objects: {object_text}"
            
            # Attention visualization
            attention_image = None
            if show_attention:
                try:
                    attention_weights = self.model.get_attention_weights(
                        image_tensor, input_ids, attention_mask
                    )
                    
                    # Convert image to numpy for visualization
                    image_np = np.array(image.resize((224, 224)))
                    
                    # Get text tokens
                    tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
                    tokens = [t for t in tokens if t != "[PAD]"]
                    
                    # Create attention visualization
                    attention_image = visualize_attention(
                        image_np, attention_weights[0], tokens, save_path=None
                    )
                except Exception as e:
                    print(f"Error creating attention visualization: {e}")
        
        return scene_prediction, object_text, description, attention_image
    
    def create_demo_interface(self) -> gr.Blocks:
        """Create the Gradio demo interface.
        
        Returns:
            Gradio Blocks interface.
        """
        with gr.Blocks(
            title="Multi-Modal Scene Understanding",
            theme=gr.themes.Soft()
        ) as demo:
            gr.Markdown(
                """
                # Multi-Modal Scene Understanding Demo
                
                This demo showcases a multi-modal AI model that combines visual and textual information 
                to understand scenes. Upload an image and provide a text description to see how the model 
                analyzes the scene, detects objects, and generates insights.
                
                **Features:**
                - Scene type classification (indoor/outdoor)
                - Object detection and recognition
                - Cross-modal attention visualization
                - Natural language scene description
                """
            )
            
            with gr.Row():
                with gr.Column():
                    # Input components
                    image_input = gr.Image(
                        label="Upload Image",
                        type="pil",
                        height=300
                    )
                    
                    text_input = gr.Textbox(
                        label="Scene Description",
                        placeholder="Describe what you see in the image...",
                        lines=3
                    )
                    
                    show_attention = gr.Checkbox(
                        label="Show Attention Visualization",
                        value=True
                    )
                    
                    predict_btn = gr.Button("Analyze Scene", variant="primary")
                
                with gr.Column():
                    # Output components
                    scene_output = gr.Textbox(
                        label="Scene Type Prediction",
                        interactive=False
                    )
                    
                    objects_output = gr.Textbox(
                        label="Detected Objects",
                        interactive=False
                    )
                    
                    description_output = gr.Textbox(
                        label="Generated Description",
                        interactive=False
                    )
                    
                    attention_output = gr.Image(
                        label="Attention Visualization",
                        height=300
                    )
            
            # Example inputs
            gr.Examples(
                examples=[
                    ["A person sitting on a chair in a living room with a cat on the sofa"],
                    ["Two people walking in a park with trees and a dog"],
                    ["A kitchen with a person cooking and dishes on the counter"],
                    ["A beach scene with people swimming and boats in the water"]
                ],
                inputs=[text_input],
                label="Example Descriptions"
            )
            
            # Event handlers
            predict_btn.click(
                fn=self.predict_scene,
                inputs=[image_input, text_input, show_attention],
                outputs=[scene_output, objects_output, description_output, attention_output]
            )
            
            # Safety disclaimer
            gr.Markdown(
                """
                ## Disclaimer
                
                This is a research/educational demo. The model may not be accurate for all images and scenarios. 
                Do not use this model for critical applications without proper validation.
                
                **Limitations:**
                - Model performance depends on training data
                - May not generalize to all scene types
                - Object detection accuracy varies by image quality
                - Text descriptions are generated based on limited context
                """
            )
        
        return demo


def main():
    """Main function to launch the demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch scene understanding demo")
    parser.add_argument("--config", type=str, default="configs/model/default.yaml",
                       help="Path to model configuration")
    parser.add_argument("--port", type=int, default=7860,
                       help="Port to run the demo on")
    parser.add_argument("--share", action="store_true",
                       help="Create a public link")
    
    args = parser.parse_args()
    
    # Create demo
    demo_app = SceneUnderstandingDemo(args.config)
    demo = demo_app.create_demo_interface()
    
    # Launch demo
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    main()
