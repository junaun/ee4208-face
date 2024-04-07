# ee4208-face
Face Detection and Recognition System, EE4208 Group1 Assignment 1

## How to Run
1. Create and activate Conda Environment(Recommended)
2. Clone or download this repo
3. Run `pip install -r requirements.txt`
4. Run `python face_recognition.py`

## To add your own face data
1. Create a folder named `face_image`
2. Create a folder with your name inside `face_image`
3. Add your face image of any size inside the folder and name it like this 'johndoe (1).jpg` (separated by a ' ')
4. Remove the known_embeddings.pt and known_names.pt every time you add or remove data

### References
1. Tim Esler's facenet repo in PyTorch : https://github.com/timesler/facenet-pytorch
