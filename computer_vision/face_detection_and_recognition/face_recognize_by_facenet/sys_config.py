"""
Created on Thu Jan 17 14:04:02 2019

system config

@author: zyb_as
"""

# TODO:
facenet_weight_dir = 'Absolute path of face_recognize_by_facenet/facenet_pretrained_model'
template_face_embeddings_file = 'Absolute path of face_recognize_by_facenet/template/template_face_embeddings.npy'

# minimum size of face to be detected
face_minsize = 60 

# resize the detected face into uniform size
image_size = 160

# the margin between face and bound
margin = 44

# three steps's threshold (read mtcnn paper to find the meaning)
three_threshold = [ 0.6, 0.7, 0.7 ]  

# scale factor (read mtcnn paper to find the meaning)
factor = 0.709 

# the distance threshold between two face's embeddings
distance_threshold = 0.52

