# Import necessary libraries
import gradio as gr  # For creating a user interface
import easyocr  # EasyOCR for text recognition
import pytesseract  # Tesseract OCR engine
import cv2  # OpenCV for image processing
import pandas as pd  # Pandas for handling tabular data
from PIL import Image  # Image manipulation
import numpy as np  # Array manipulations
import os  # File path and OS utilities
from typing import Tuple, List, Dict  # Type hinting

class OCRProcessor:
    def __init__(self):
        # Configure Tesseract path for Windows
        tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(tesseract_path):  # Check if Tesseract is installed
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        # Initialize EasyOCR reader placeholder
        self.reader = None

    def initialize_easyocr(self, use_gpu: bool):
        """Initialize EasyOCR reader with GPU preference."""
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)  # English language OCR

    def process_easyocr(self, image: np.ndarray, detail_level: int) -> Tuple[List, pd.DataFrame]:
        """
        Perform OCR using EasyOCR.
        :param image: Input image in numpy array format
        :param detail_level: Determines the verbosity of OCR results
        :return: OCR results and a DataFrame with text, confidence, and bounding boxes
        """
        results = self.reader.readtext(image, detail=detail_level)
        # Structure the results into a DataFrame
        data = [
            {
                'Text': text, 
                'Confidence': conf, 
                'Bounding Box': [tuple(map(int, pt)) for pt in bbox]
            } 
            for bbox, text, conf in results
        ]
        return results, pd.DataFrame(data)

    def process_tesseract(self, image: np.ndarray, conf_threshold: int) -> Tuple[List, pd.DataFrame]:
        """
        Perform OCR using Tesseract.
        :param image: Input image in numpy array format
        :param conf_threshold: Minimum confidence score for text to be included
        :return: OCR results and a DataFrame with text, confidence, and bounding boxes
        """
        # Extract OCR data from Tesseract
        text_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        results, data = [], []
        for i in range(len(text_data['text'])):
            conf = int(text_data['conf'][i])
            # Filter text with confidence above the threshold
            if conf >= conf_threshold and text_data['text'][i].strip():
                x, y, w, h = (
                    text_data['left'][i], 
                    text_data['top'][i], 
                    text_data['width'][i], 
                    text_data['height'][i]
                )
                bbox = [(x, y), (x + w, y + h)]  # Bounding box coordinates
                results.append((bbox, text_data['text'][i], conf))
                data.append({
                    'Text': text_data['text'][i], 
                    'Confidence': conf, 
                    'Bounding Box': bbox
                })
        
        return results, pd.DataFrame(data)

    def draw_bounding_boxes(self, image: np.ndarray, results: List, color: Tuple[int, int, int]) -> Image:
        """
        Draw bounding boxes around detected text.
        :param image: Input image
        :param results: OCR results with bounding box data
        :param color: Bounding box color in BGR
        :return: Image with drawn bounding boxes
        """
        image_copy = image.copy()
        for result in results:
            bbox = result[0]  # Extract bounding box coordinates
            # Convert EasyOCR quadrilateral to rectangle if necessary
            if isinstance(bbox, list) and len(bbox) == 4:
                x_min = min(int(pt[0]) for pt in bbox)
                y_min = min(int(pt[1]) for pt in bbox)
                x_max = max(int(pt[0]) for pt in bbox)
                y_max = max(int(pt[1]) for pt in bbox)
                cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), color, 2)
            else:
                # Tesseract bounding boxes are rectangular by default
                x1, y1 = tuple(map(int, bbox[0]))
                x2, y2 = tuple(map(int, bbox[1]))
                cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
        return Image.fromarray(image_copy)

def create_ocr_interface():
    processor = OCRProcessor()  # Initialize OCRProcessor instance

    def process_ocr(image, use_gpu, detail_level, conf_threshold):
        """
        Process the input image using EasyOCR and Tesseract.
        :param image: Uploaded image
        :param use_gpu: GPU usage for EasyOCR
        :param detail_level: Detail level for EasyOCR results
        :param conf_threshold: Confidence threshold for Tesseract
        :return: Processed images, data, and file paths
        """
        if image is None:
            return [None] * 6

        image_array = np.array(image)

        # Initialize EasyOCR with GPU setting
        processor.initialize_easyocr(use_gpu)

        # OCR using EasyOCR
        easy_results, easy_data = processor.process_easyocr(image_array, detail_level)
        easy_image = processor.draw_bounding_boxes(image_array, easy_results, (0, 0, 255))

        # OCR using Tesseract
        tess_results, tess_data = processor.process_tesseract(image_array, conf_threshold)
        tess_image = processor.draw_bounding_boxes(image_array, tess_results, (0, 255, 0))

        # Save results as CSV files
        easy_file_path = "easyocr_results.csv"
        tess_file_path = "tesseract_results.csv"
        easy_data.to_csv(easy_file_path, index=False)
        tess_data.to_csv(tess_file_path, index=False)

        return easy_image, tess_image, easy_data, tess_data, easy_file_path, tess_file_path

    with gr.Blocks() as interface:
        # Title of the app
        gr.Markdown("""# OCR with EasyOCR and Tesseract""")

        # Input image
        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload Image")
            
        # Configuration options
        with gr.Row():
            use_gpu = gr.Checkbox(label="Use GPU for EasyOCR", value=False)
            detail_level = gr.Slider(0, 1, step=1, label="EasyOCR Detail Level (0: Basic, 1: Verbose)", value=1)
            conf_threshold = gr.Slider(0, 100, step=1, label="Tesseract Confidence Threshold", value=30)
            
        # Buttons
        with gr.Row():
            extract_button = gr.Button("Extract Text")
            clear_button = gr.Button("Clear")

        # OCR results
        with gr.Row():
            easyocr_image = gr.Image(label="EasyOCR Result")
            tesseract_image = gr.Image(label="Tesseract Result")

        # Data outputs
        with gr.Row():
            easyocr_data = gr.Dataframe()
            tesseract_data = gr.Dataframe()

        # Downloadable results
        with gr.Row():
            easyocr_download = gr.File(label="EasyOCR CSV")
            tesseract_download = gr.File(label="Tesseract CSV")

        # Button actions
        extract_button.click(
            fn=process_ocr,
            inputs=[image_input, use_gpu, detail_level, conf_threshold],
            outputs=[
                easyocr_image, tesseract_image, 
                easyocr_data, tesseract_data, 
                easyocr_download, tesseract_download
            ]
        )

        clear_button.click(
            fn=lambda: [None]*6,  # Clear all outputs
            outputs=[
                image_input, easyocr_image, tesseract_image, 
                easyocr_data, tesseract_data, 
                easyocr_download, tesseract_download
            ]
        )

    return interface

# Run the OCR app
if __name__ == "__main__":
    interface = create_ocr_interface()
    interface.launch(debug=True)
