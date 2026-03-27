import torch
from torchvision import transforms
from face_alignment import align
from backbones import get_model

import os
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def detect_and_preprocess(img_path, target_size=(112, 112)):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        # Fallback: center-crop the image
        h, w = img.shape[:2]
        s = min(h, w)
        cx, cy = w // 2, h // 2
        x = max(cx - s // 2, 0)
        y = max(cy - s // 2, 0)
        face_img = img[y:y + s, x:x + s]
    else:
        # choose the largest detected face
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        face_img = img[y:y + h, x:x + w]
    
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, target_size)
    face_norm = (face_resized.astype(np.float32) - 127.5) / 128.0
    face = np.expand_dims(face_norm, axis=0)  # shape: (1, 112, 112, 3)
    return face.transpose(0, 3, 1, 2)  # shape: (1, 3, 112, 112)


# load model
model_name="edgeface_s_gamma_05" # or edgeface_xs_gamma_06
model=get_model(model_name)
checkpoint_path=f'/home/faith/edgeface/checkpoints/{model_name}.pt'
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
model.eval()

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])


def get_embed(path):
    transformed_input = detect_and_preprocess(path)
    transformed_input = torch.from_numpy(transformed_input).float()
    
    with torch.no_grad():
        # print(transformed_input.shape)
        embedding = model(transformed_input)
    return embedding.squeeze(0).cpu().numpy()


def get_embedding(path):
    # path = 'path_to_face_image'
    aligned = align.get_aligned_face(path) # align face
    if aligned is None:
        raise RuntimeError(f"Failed to align face for: {path}")
    transformed_input = transform(aligned).unsqueeze(0) # preprocessing

    # extract embedding
    with torch.no_grad():
        print(transformed_input.shape)
        embedding = model(transformed_input)
    return embedding.squeeze(0).cpu().numpy()


def cosine_similarity(embedding1, embedding2):
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        raise ValueError("Embedding norm is zero; cosine similarity is undefined")
    return float(np.dot(embedding1, embedding2) / (norm1 * norm2))


if __name__ == "__main__":
    img1_path = '/home/faith/edgeface/didi.jpg'  # replace with your image path
    # emb1 = get_embedding(img1_path)

    img2_path = '/home/faith/edgeface/me.jpg'  # replace with your image path
    # emb2 = get_embedding(img2_path)
    # print(emb1)
    # print(emb2)

    emb1 = get_embed(img1_path)
    emb2 = get_embed(img2_path)

    cosine_sim = cosine_similarity(emb1, emb2)
    print(f"Embedding norms: {np.linalg.norm(emb1):.6f}, {np.linalg.norm(emb2):.6f}")
    print(f"Cosine similarity between '{os.path.basename(img1_path)}' and '{os.path.basename(img2_path)}': {cosine_sim:.6f}")