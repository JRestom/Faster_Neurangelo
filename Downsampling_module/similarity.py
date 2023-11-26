import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import shutil
import time


start_time = time.time()
# Load pretrained model
model = models.resnet50(pretrained=True) #If too slow, consider using Mobile architectures
# Remove the final classification layer (so that we only get the feature map)
model = nn.Sequential(*list(model.children())[:-1])
model.eval()  


# Preprocess image function
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')

    #Normalize and resize the image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = preprocess(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img

# Function to extract features using the pretrained model that we defined previosly
def extract_features(image_path, model):
    img = preprocess_image(image_path)
    with torch.no_grad():
        features = model(img)
    return features.squeeze().numpy()  # Convert torch tensor to numpy array

# # Paths to your image pairs
# image_path_1 = '/home/jose.viera/projects/cv802/neuralangelo/datasets/test3_ds2/images/000001.jpg'
# image_path_2 = '/home/jose.viera/projects/cv802/neuralangelo/datasets/test3_ds2/images/000060.jpg'              
# Extract features for both images using the pre-trained ResNet-50 model
# features_1 = extract_features(image_path_1, model)
# features_2 = extract_features(image_path_2, model)
# Compute cosine similarity between the feature vectors (We can also test using other similarity measures)
#This metric is in range 0-1 (1 being the most similar)
# similarity_score = cosine_similarity([features_1], [features_2])[0][0]
# print(f"Similarity score between the images: {similarity_score}")

# Path to the folder containing images
folder_path = '/home/jose.viera/projects/cv802/neuralangelo/datasets/test3_ds2/images'

# List all image files in the folder
image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path)
               if file.endswith(('jpg'))]

image_files.sort()





# Compute pairwise similarity between images until similarity drops below threshold

threshold = 0.95  # Adjust this threshold as needed
below_threshold_images = []

# Ensure the first image is added to below_threshold_images
below_threshold_images.append(image_files[0])

#Start features as the features of the first image
avg_features = extract_features(image_files[0], model)



anchor = 0
while anchor < len(image_files):
    features_1 = extract_features(image_files[anchor], model) #Features anchor
    

    found_new_anchor = False
    for i, img_path in enumerate(image_files[anchor+1:], start=anchor+1):

        features_2 = extract_features(img_path, model) #Features new image
        similarity_score = cosine_similarity([features_1], [features_2])[0][0] 

        if similarity_score < threshold:
            print(f"Found new anchor at position: {i}")
            print(similarity_score)

            #Compute distant to average of features
            similarity_to_avg = cosine_similarity([avg_features], [features_2])[0][0] 

            if similarity_to_avg>0.74:
                
                # Incrementally update the average features
                num_saved = len(below_threshold_images)
                avg_features = ((num_saved) * avg_features + features_2) / num_saved+1

                below_threshold_images.append(img_path)
                anchor = i
                found_new_anchor = True
                break #Exit inner loop
                
            else:
                print('Similarity to avg not acceptable')
                continue

            
    if not found_new_anchor:
        break  # Exit outer loop if no new anchor found


   


print(f"Number of original images: {len(image_files)}")
print(f"Number of downsampled images: {len(below_threshold_images)}")

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

start_time = time.time()

# Array of image paths
image_paths = below_threshold_images

# Create a new folder to save the copied images
output_folder = '/home/jose.viera/projects/cv802/neuralangelo/datasets/downsampled'
os.makedirs(output_folder, exist_ok=True)

# Copy images to the new folder
for img_path in image_paths:
    if os.path.exists(img_path):  # Check if the image path exists
        img_name = os.path.basename(img_path)  # Get the image file name
        output_path = os.path.join(output_folder, img_name)  # Destination path
        shutil.copyfile(img_path, output_path)  # Copy the image to the new folder
        print(f"Copied {img_name} to {output_folder}")
    else:
        print(f"Image path {img_path} does not exist.")

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time for copying: {execution_time} seconds")