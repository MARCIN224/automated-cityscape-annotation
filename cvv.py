import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
import torch
import torchvision
import os
from segment_anything import SamPredictor, sam_model_registry

class SimpleCityscapeDecomposer:
    def __init__(self, sam_checkpoint_path="sam_vit_h_4b8939.pth", model_size="large"):
        print("Loading models... This may take a minute.")
        print(f"Using SAM model: {model_size.upper()} ({sam_checkpoint_path})")
        
        self.models_loaded = False
        
        if not os.path.exists(sam_checkpoint_path):
            print(f"SAM model file not found: {sam_checkpoint_path}")
            return
        
        self.device = self._setup_device()
        print(f"Using device: {self.device}")
        
        try:
            print("Loading SAM model...")
            if model_size == "large":
                model_type = "vit_h"
            elif model_size == "medium":
                model_type = "vit_l" 
            else:
                model_type = "vit_b"
                
            self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
            self.sam.to(device=self.device)
            self.predictor = SamPredictor(self.sam)
            
            print("Loading DeepLabV3...")
            self.semantic_model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True, verbose=False)
            self.semantic_model.eval().to(self.device)
            
            print("Loading object detection model...")
            self.detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, verbose=False)
            self.detection_model.eval().to(self.device)
            
            print("All models loaded successfully!")
            self.models_loaded = True
            self.model_size = model_size
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.models_loaded = False
    
    def _setup_device(self):
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            try:
                torch.cuda.init()
                device = torch.device('cuda')
                test_tensor = torch.tensor([1.0]).cuda()
                print("GPU is available and working!")
                return device
            except Exception as e:
                print(f"GPU detected but failed to initialize: {e}")
                return torch.device('cpu')
        else:
            print("CUDA not available, using CPU")
            return torch.device('cpu')
    
    def get_enhanced_edge_layer(self, image):
        if not self.models_loaded:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_visualization = np.zeros_like(image)
            edge_visualization[edges > 0] = [255, 255, 255]
            return edge_visualization
            
        try:
            print("Using SAM for edge detection...")
            self.predictor.set_image(image)
            
            image_embedding = self.predictor.get_image_embedding()
            
            masks, scores, logits = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=None,
                multimask_output=False,
            )
            
            mask = masks[0]
            edges = (cv2.Canny((mask * 255).astype(np.uint8), 50, 150) > 0)
            
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            traditional_edges = cv2.Canny(gray, 30, 100)
            
            combined_edges = np.logical_or(edges, traditional_edges > 0)
            
            edge_visualization = np.zeros_like(image)
            edge_visualization[combined_edges] = [255, 255, 255]
            
            return edge_visualization
            
        except Exception as e:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_visualization = np.zeros_like(image)
            edge_visualization[edges > 0] = [255, 255, 255]
            return edge_visualization
    
    def get_enhanced_stuff_layer(self, image):
        if not self.models_loaded:
            return self.get_basic_stuff_layer(image)
            
        try:
            self.predictor.set_image(image)
            
            h, w = image.shape[:2]
            points = []
            for y in range(100, h, 100):
                for x in range(100, w, 100):
                    points.append([x, y])
            
            if points:
                points = np.array(points)
                masks, scores, logits = self.predictor.predict(
                    point_coords=points,
                    point_labels=np.ones(len(points)),
                    multimask_output=True,
                )
                
            return self.get_basic_stuff_layer(image)
            
        except:
            return self.get_basic_stuff_layer(image)
    
    def get_basic_stuff_layer(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        height, width = image.shape[:2]
        stuff_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        sky_mask = cv2.inRange(hsv, (90, 40, 40), (140, 255, 255))
        stuff_mask[sky_mask > 0] = [255, 0, 0]
        
        green_mask = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
        stuff_mask[green_mask > 0] = [0, 255, 0]
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        road_mask = (gray < 120) & (gray > 10)
        stuff_mask[road_mask] = [128, 128, 128]
        
        building_mask = cv2.inRange(hsv, (0, 0, 50), (180, 50, 200))
        stuff_mask[building_mask > 0] = [0, 0, 255]
        
        return cv2.addWeighted(image, 0.6, stuff_mask, 0.4, 0)
    
    def get_things_layer(self, image):
        if not self.models_loaded:
            return image
            
        result = image.copy()
        
        try:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.to(self.device)
            
            with torch.no_grad():
                predictions = self.detection_model([image_tensor])
            
            class_names = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter'
            ]
            
            boxes = predictions[0]['boxes'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()
            
            for box, score, label in zip(boxes, scores, labels):
                if score > 0.5 and label <= len(class_names):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    class_name = class_names[label-1] if label-1 < len(class_names) else f'class{label}'
                    
                    
                    text = f'{class_name}: {score:.2f}'
                    text_position = (x1, y1-10)
                    
                    
                    cv2.putText(result, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    
                    cv2.putText(result, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        except Exception as e:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 1000 > area > 100:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    
                    text = 'object'
                    text_position = (x, y-10)
                    
                    
                    cv2.putText(result, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    
                    cv2.putText(result, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result
    
    def decompose_image(self, image_path):
        if not self.models_loaded:
            print("Models not loaded properly. Cannot process image.")
            return
        
        print(f"Processing image: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Could not load image!")
            return
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print("Generating Stuff Layer...")
        stuff_layer = self.get_enhanced_stuff_layer(image)
        
        print("Generating Things Layer...")
        things_layer = self.get_things_layer(image)
        
        print("Generating Edge Layer...")
        edge_layer = self.get_enhanced_edge_layer(image)
        
        self.display_results(image, stuff_layer, things_layer, edge_layer)
    
    def display_results(self, original, stuff, things, edges):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        
        plt.subplots_adjust(
            left=0.013,
            bottom=0.015,
            right=0.987,
            top=0.968,
            wspace=0.032,
            hspace=0.067
        )
        
        plots = [
            (original, 'Original Image'),
            (stuff, 'Stuff Layer'),
            (things, 'Detected Objects'),
            (edges, 'Edge Detection')
        ]
        
        for i, (img, title) in enumerate(plots):
            ax = axes[i // 2, i % 2]
            ax.imshow(img)
            
            
            ax.set_title(title, fontweight='bold', fontsize=11, color='white',
                        path_effects=[withStroke(linewidth=2, foreground='black')])
            ax.axis('off')
    
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    decomposer = SimpleCityscapeDecomposer(
        sam_checkpoint_path="sam_vit_h_4b8939.pth", 
        model_size="large"
    )
    
    if decomposer.models_loaded:
        image_path = r""   #Paste your image location here
        
        if os.path.exists(image_path):
            print(f"Image found: {image_path}")
            decomposer.decompose_image(image_path)
        else:
            print(f"Image not found: {image_path}")
    else:
        print("Failed to initialize with large model.")
