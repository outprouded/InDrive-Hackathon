
# Car Integrity Pipeline Documentation

## Overview
The **Car Integrity Pipeline** is a Streamlit-based web application designed for vehicle inspection, performing damage detection, cleanliness classification, and dirt severity assessment on car images. It uses deep learning models (YOLOv8, ViT, and VGG) to analyze images and provide detailed results.

This documentation explains how to use the Streamlit interface, configure settings, and interpret outputs.

## Requirements
- **Python**: 3.8 or higher
- **Dependencies**: Install required packages using:
  ```bash
  pip install streamlit torch torchvision ultralytics timm numpy pillow
  ```
- **Model Weights**: Ensure model weights are available at the paths specified in the `CFG` dictionary in `app.py`. Update paths if necessary.
- **Hardware**: A CUDA-enabled GPU is recommended for faster inference, but CPU is supported.

## Running the Application
1. Save the application code as `app.py`.
2. Open a terminal in the directory containing `app.py`.
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Open the provided URL (typically `http://localhost:8501`) in a web browser.

## Interface Overview
The interface is divided into:
- **Main Panel**: Displays the title, image upload/folder input, and analysis results.
- **Sidebar**: Contains configuration options for model selection, thresholds, and display settings.
- **Results Section**: Shows analysis outputs for each image, including damage, cleanliness, detection, and severity results.

### Main Components
- **Title**: "Car Integrity Analysis" with a description of the pipeline's purpose.
- **Image Input**: Allows uploading images or specifying a folder path.
- **Analysis Results**: Displays processed images, classification results, and optional probability tables/detection details.
- **Sidebar**: Configures model choices, thresholds, and display options.

## Step-by-Step Usage

### 1. Configure Settings in the Sidebar
The sidebar is organized into expandable sections for ease of use.

#### Model Selection
- **Damage Classifier**: Choose between:
  - **YOLOv8s-cls**: A YOLOv8 small model for damage classification.
  - **ViT (visual transformer)**: A Vision Transformer model for damage classification.
- **Severity Classifier**: Choose between:
  - **YOLOv8l-cls**: A YOLOv8 large model for dirt severity classification.
  - **VGG (custom)**: A VGG model for dirt severity classification.

#### Decision Thresholds
Adjust sliders to set confidence thresholds for triggering subsequent analyses:
- **Trigger detection if 'Damaged' confidence ≥**: Default is 0.50. Higher values require stronger confidence to run damage detection.
- **Trigger severity if 'Dirty' confidence ≥**: Default is 0.50. Higher values require stronger confidence to run severity classification.
- **Detection Confidence**: Default is 0.25. Sets the minimum confidence for detected damage types.
- **Detection IoU (NMS)**: Default is 0.70. Controls non-maximum suppression for overlapping detections.

#### Display Options
- **Show Probability Tables**: Check to display probability distributions for damage, cleanliness, and severity classifications.
- **Show Detection JSON**: Check to display raw detection results in JSON format (useful for debugging or detailed analysis).

#### Additional Information
- **Device**: Displays whether the app is running on CPU or GPU (e.g., `cuda:0`).
- **Logic**: Summarizes the pipeline logic: always classify damage and cleanliness; run detection if damaged, and severity if dirty.

### 2. Upload Images or Specify a Folder
- **Upload Images**:
  1. In the main panel, use the file uploader to select one or more images (supported formats: JPG, JPEG, PNG, BMP, WebP).
  2. Multiple images can be uploaded simultaneously.
- **Specify Folder Path**:
  1. Enter a valid folder path containing images (e.g., `C:/images/`).
  2. The app will load all supported images from the folder.
- If no images are provided or the folder path is invalid, a message will prompt you to upload images or correct the path.

### 3. Run the Pipeline
Once images are uploaded or a valid folder path is provided:
1. The app loads the selected models (indicated by a "Loading models..." spinner).
2. A progress bar tracks the processing of each image.
3. Results are displayed for each image in the "Analysis Results" section.

### 4. Interpret Results
For each image, the results are displayed in two columns:
- **Left Column**: Shows the input image.
- **Right Column**: Displays classification results, detection outputs (if applicable), and optional probability tables/JSON.

#### Output Details
- **Damage Classification**: Indicates whether the car is "damaged" or "undamaged" with confidence score (e.g., `damaged (0.892)`).
- **Cleanliness Classification**: Indicates whether the car is "clean" or "dirty" with confidence score (e.g., `dirty (0.765)`).
- **Detection (if triggered)**: If the damage confidence exceeds the threshold, a detection image with bounding boxes is shown, highlighting detected damage types (e.g., dent, scratch).
- **Severity (if triggered)**: If the cleanliness is "dirty" and confidence exceeds the threshold, severity is classified as "slightly_dirty", "dirty", or "very_dirty" with a confidence score.
- **Probability Tables** (optional): Shows probability distributions for each classification (damage, cleanliness, severity).
- **Detection JSON** (optional): Displays raw detection data, including labels, confidences, and bounding box coordinates.

#### Visual Feedback
- **Spinners**: Indicate when models are loading or inference is running.
- **Progress Bar**: Shows the progress of processing multiple images.
- **Success Message**: Confirms successful model loading and analysis completion.

### 5. Review and Adjust
- Adjust thresholds or model selections in the sidebar and re-upload images to refine results.
- Toggle display options to show/hide probability tables or JSON outputs as needed.

## Example Workflow
1. Open the app in your browser.
2. In the sidebar:
   - Select "ViT (visual transformer)" for damage classification.
   - Select "VGG (custom)" for severity classification.
   - Set damage threshold to 0.60 and dirty threshold to 0.55.
   - Enable "Show Probability Tables" and disable "Show Detection JSON".
3. Upload two car images (e.g., `car1.jpg`, `car2.png`).
4. Wait for the app to process the images (progress bar updates).
5. Review results:
   - For `car1.jpg`: "Damage: damaged (0.920) | Cleanliness: dirty (0.780) → Detection ran → Severity: very_dirty (0.850)".
   - Detection image shows dents and scratches.
   - Probability tables show confidence distributions.
6. Adjust thresholds and re-upload if needed.

## Troubleshooting
- **Model Loading Errors**: Ensure weight file paths in `CFG` are correct and accessible.
- **Image Reading Errors**: Verify uploaded images are in supported formats and not corrupted.
- **Performance Issues**: If running on CPU, inference may be slow. Consider using a GPU or reducing image resolution.
- **No Results**: Ensure images are uploaded or the folder path is valid. Check the console for error messages.

## Notes
- The app uses a modern, clean design with rounded corners, a light background, and consistent typography for better usability.
- Expanders in the sidebar organize settings to reduce clutter.
- The pipeline logic ensures efficient processing by only running detection and severity classification when thresholds are met.
- For large datasets, process images in smaller batches to avoid memory issues.

## Support
For issues or questions, refer to the Streamlit documentation or check the console output for detailed error messages. Ensure all dependencies are installed and model weights are correctly configured.

## Models and Datasets
[Models and Datasets](https://drive.google.com/drive/folders/1vzkzQ6-6ZQPCHrWP76h7jX_1WCoiN21_?usp=sharing)
