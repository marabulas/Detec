import torch
import cv2
import numpy as np
from PIL import Image

# Load the pre-trained DeepFashion2 model
model = torch.hub.load('pytorch/vision:v0.9.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()

# Define the class labels
classes = ['background', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes', 'dress', 'coat', 'socks', 'pants', 'jumpsuits', 'scarf', 'skirt', 'leftShoe', 'rightShoe']


# Load an image and convert it to a PyTorch tensor
image = Image.open('image.jpg')
image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).float()

# Move the tensor to the GPU if available
if torch.cuda.is_available():
    device = torch.device('cuda')
    model.to(device)
    image_tensor = image_tensor.to(device)

# Perform the inference
with torch.no_grad():
    output = model(image_tensor)['out'][0]

# Convert the output tensor to a numpy array
output = output.detach().cpu().numpy()

# Convert the output to a binary mask by thresholding at 0.5
mask = (output > 0.5).astype(np.uint8)

# Get the bounding boxes, confidence scores, and class labels for the clothing items
boxes = []
scores = []
labels = []
for i in range(1, len(classes)):
    contours, _ = cv2.findContours(mask[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append([x, y, x+w, y+h])
        scores.append(output[i, y:y+h, x:x+w].mean())
        labels.append(classes[i])

# Draw the bounding boxes, confidence scores, and class labels on the image
image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
for box, score, label in zip(boxes, scores, labels):
    if score > 0.5:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(image, f'{label}: {score:.2f}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image
cv2.imshow('Detected Clothing Items', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
