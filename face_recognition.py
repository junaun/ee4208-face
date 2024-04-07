from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import cv2 as cv
from PIL import Image
import os
from tqdm import tqdm
import time

# help(MTCNN)
# help(InceptionResnetV1)

def draw_faces(image, bounding_box, name):
    cv.rectangle(image,
                 (int(bounding_box[0]), int(bounding_box[1])),
                 (int(bounding_box[2]), int(bounding_box[3])),
                 (0, 155, 255),
                 2)
    cv.putText(image, name, (int(bounding_box[0]), int(bounding_box[1]) - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)

def preprocess_image(image_path):
    image = Image.open(image_path)
    image_cropped = mtcnn(image)
    if image_cropped is not None:
        return resnet(image_cropped.unsqueeze(0))
    return None

def find_closest_embeddings(embedding):
    normalized_known_embeddings = torch.nn.functional.normalize(known_embeddings, p=2, dim=1)
    normalized_embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    
    cosine_similarities = torch.matmul(normalized_embedding, normalized_known_embeddings.t())
    max_similarity, max_idx = torch.max(cosine_similarities, dim=1)
    
    if max_similarity >= 0.65:  # Adjust this threshold as needed
        return known_names[max_idx]
    return "Unknown"

mtcnn = MTCNN(select_largest=False, margin=0, device='cuda' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2', classify=False).eval().to('cuda' if torch.cuda.is_available() else 'cpu')

known_embeddings = []
known_names = []

image_dir = "./face_image/"
embeddings_path = "known_embeddings.pt"
names_path = "known_names.pt"

if os.path.exists(embeddings_path) and os.path.exists(names_path):
    # Load embeddings and names
    known_embeddings = torch.load(embeddings_path)
    known_names = torch.load(names_path)
else:
    # Iterate over each person's directory and preprocess each image
    for person_name in tqdm(os.listdir(image_dir)):
        person_dir = os.path.join(image_dir, person_name)
        if os.path.isdir(person_dir):  # Check if it's a directory
            person_embeddings = []
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                embedding = preprocess_image(image_path)
                if embedding is not None:
                    person_embeddings.append(embedding)
            # Average the embeddings for the current person
            if person_embeddings:
                person_embedding = torch.mean(torch.cat(person_embeddings, dim=0), dim=0, keepdim=True)
                known_embeddings.append(person_embedding)
                known_names.append(person_name)
    torch.save(known_embeddings, embeddings_path)
    torch.save(known_names, names_path)

known_embeddings = torch.cat(known_embeddings, dim=0)

cap = cv.VideoCapture(0)
# cap = cv.VideoCapture("video.mp4")

fps = 0
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()

    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time 
    cv.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if not ret:
        print("Failed to grab frame")
        break

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    
    # Detect faces
    boxes, _ = mtcnn.detect(frame_pil)
    
    if boxes is not None:
        for box in boxes:
            try:
                # Ensure the crop is a valid operation
                face = frame_pil.crop(box)
                face_tensor = mtcnn(face)
                if face_tensor is not None:
                    embedding = resnet(face_tensor.unsqueeze(0))
                    name = find_closest_embeddings(embedding)
                    draw_faces(frame, box, name)
            except RuntimeError as e:
                print(f"Skipping a frame due to an error: {e}")

    cv.imshow('Frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
