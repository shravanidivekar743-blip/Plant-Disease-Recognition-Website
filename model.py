import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io

# 1️⃣ Model class
class Plant_Disease_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Pretrained ResNet34
        self.network = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        # Manually set fc layer for 38 classes
        self.network.fc = nn.Linear(self.network.fc.in_features, 38)

    def forward(self, xb):
        return self.network(xb)

# 2️⃣ Transform
transform = transforms.Compose([
    transforms.Resize(size=128),
    transforms.ToTensor()
])

# 3️⃣ Class names
num_classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# 4️⃣ Initialize model
model = Plant_Disease_Model()

# 5️⃣ Load pretrained weights (legacy .pth, DO NOT use weights_only)
checkpoint = torch.load('./Models/plantDisease-resnet34.pth', map_location='cpu', weights_only=False)

# 6️⃣ Copy weights except fc layer
new_state_dict = {}
for k, v in checkpoint.items():
    if "fc.weight" in k or "fc.bias" in k:
        continue  # Ignore last layer
    new_state_dict["network." + k] = v

# 7️⃣ Load weights into model
model.load_state_dict(new_state_dict, strict=False)
model.eval()  # Set model to evaluation mode

# 8️⃣ Prediction function
def predict_image(img):
    img_pil = Image.open(io.BytesIO(img))
    tensor = transform(img_pil)
    xb = tensor.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return num_classes[preds[0].item()]