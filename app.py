import os
import cv2
import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from PIL import Image
from torch.nn import functional as F
from flask import Flask, render_template, request, url_for, redirect, send_from_directory
from werkzeug.utils import secure_filename

# Model Building: Defining Blocks for the Generator Network

# Residual Block to create skip connections and better gradient flow
class ResBlock(nn.Module):
    def __init__(self, num_channel):
        super(ResBlock, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(num_channel, num_channel, 3, 1, 1),  # Convolution Layer 1
            nn.BatchNorm2d(num_channel),                    # Batch Normalization
            nn.ReLU(inplace=True),                          # ReLU Activation
            nn.Conv2d(num_channel, num_channel, 3, 1, 1),  # Convolution Layer 2
            nn.BatchNorm2d(num_channel))                   # Batch Normalization
        self.activation = nn.ReLU(inplace=True)           # Final Activation

    def forward(self, inputs):
        # Adding skip connection and applying ReLU
        output = self.conv_layer(inputs)
        output = self.activation(output + inputs)         # Skip connection
        return output


# Downsampling Block for the Encoder Section
class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownBlock, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 2, 1),   # Convolution Layer with Stride 2 (downsampling)
            nn.BatchNorm2d(out_channel),                    # Batch Normalization
            nn.ReLU(inplace=True),                          # ReLU Activation
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),   # Convolution Layer 2
            nn.BatchNorm2d(out_channel),                    # Batch Normalization
            nn.ReLU(inplace=True))                          # ReLU Activation

    def forward(self, inputs):
        output = self.conv_layer(inputs)                  # Forward Pass
        return output


# Upsampling Block for the Decoder Section
class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, is_last=False):
        super(UpBlock, self).__init__()
        self.is_last = is_last
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),     # Convolution Layer 1
            nn.BatchNorm2d(in_channel),                      # Batch Normalization
            nn.ReLU(inplace=True),                           # ReLU Activation
            nn.Upsample(scale_factor=2),                     # Upsampling Layer
            nn.Conv2d(in_channel, out_channel, 3, 1, 1))     # Convolution Layer 2
        self.act = nn.Sequential(
            nn.BatchNorm2d(out_channel),                     # Batch Normalization
            nn.ReLU(inplace=True))                           # ReLU Activation
        self.last_act = nn.Tanh()                           # For final output to match pixel range [-1, 1]

    def forward(self, inputs):
        output = self.conv_layer(inputs)
        if self.is_last:
            output = self.last_act(output)                 # Apply Tanh for final output
        else:
            output = self.act(output)                      # Apply ReLU activation for intermediate layers
        return output


# Simple Generator Network combining the above blocks
class SimpleGenerator(nn.Module):
    def __init__(self, num_channel=32, num_blocks=4):
        super(SimpleGenerator, self).__init__()
        # Downsampling layers (Encoder)
        self.down1 = DownBlock(3, num_channel)
        self.down2 = DownBlock(num_channel, num_channel*2)
        self.down3 = DownBlock(num_channel*2, num_channel*3)
        self.down4 = DownBlock(num_channel*3, num_channel*4)
        # Residual Blocks for more complex features
        res_blocks = [ResBlock(num_channel*4)]*num_blocks
        self.res_blocks = nn.Sequential(*res_blocks)
        # Upsampling layers (Decoder)
        self.up1 = UpBlock(num_channel*4, num_channel*3)
        self.up2 = UpBlock(num_channel*3, num_channel*2)
        self.up3 = UpBlock(num_channel*2, num_channel)
        self.up4 = UpBlock(num_channel, 3, is_last=True)  # Output layer (3 channels for RGB)

    def forward(self, inputs):
        # Encoder Pass
        down1 = self.down1(inputs)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        # Residual Processing
        down4 = self.res_blocks(down4)
        # Decoder Pass with skip connections
        up1 = self.up1(down4)
        up2 = self.up2(up1+down3)  # Adding skip connections
        up3 = self.up3(up2+down2)
        up4 = self.up4(up3+down1)
        return up4


# Flask Web App to Handle Image Upload and Cartoonization
UPLOAD_FOLDER = 'static/uploads/'           # Folder for uploaded files
DOWNLOAD_FOLDER = 'static/downloads/'       # Folder for processed files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}  # Allowed image formats

app = Flask(__name__, static_url_path="/static")
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
# Set the max file upload size to 10MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Function to check allowed file extensions
def allowed_file(filename):
   return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Index route: Homepage with image upload form
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file attached in the request')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print("No file selected..")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))  # Save the uploaded file
            process_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), filename)
            return redirect(url_for('uploaded_file', filename=filename))
    return render_template('index.html')  # Render the homepage template

# Function to process the uploaded image
def process_file(path, filename):
    cartoonize(path, filename)

# Cartoonize function for image transformation using the model
def cartoonize(path, filename):
    # Load the pre-trained model weights
    weight = torch.load('weight.pth', map_location='cpu')
    model = SimpleGenerator()  # Create the model instance
    model.load_state_dict(weight)  # Load weights into the model
    model.eval()  # Set the model to evaluation mode (no gradient updates)
    
    # Open the input image and preprocess it
    image = Image.open(path)
    new_image = image.resize((256, 256))  # Resize the image for model input
    new_image.save(path)  # Save the resized image back

    # Prepare the image for the model
    raw_image = cv2.imread(path)
    image = raw_image / 127.5 - 1  # Normalize the image to [-1, 1] range
    image = image.transpose(2, 0, 1)  # Convert HWC -> CHW format (for PyTorch)
    image = torch.tensor(image).unsqueeze(0)  # Add batch dimension

    # Pass the image through the model
    output = model(image.float())
    output = output.squeeze(0).detach().numpy()  # Convert tensor to numpy array
    output = output.transpose(1, 2, 0)  # Convert back to HWC format
    output = (output + 1) * 127.5  # Rescale back to [0, 255]
    output = np.clip(output, 0, 255).astype(np.uint8)  # Ensure valid pixel values

    # Combine the original and cartoonized images side by side
    output = np.concatenate([raw_image, output], axis=1)
    
    # Save the resulting image
    cv2.imwrite(os.path.join(app.config['DOWNLOAD_FOLDER'], filename), output)

# Route to serve the processed image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    # Run the Flask app on the specified port
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)  # Expose the app publicly for testing
