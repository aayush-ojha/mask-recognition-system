import cv2
import numpy as np
import tensorflow as tf

def load_model(model_path):
    graph = tf.Graph()
    with graph.as_default():
        with tf.compat.v1.Session() as sess:
            with tf.io.gfile.GFile(model_path, "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")
    return graph

def detect_mask(sess, graph, frame, face_cascade, input_tensor, output_tensor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (260, 260)) 
        face = np.expand_dims(face, axis=0)
        face = face / 255.0  

        
        prediction = sess.run(output_tensor, feed_dict={input_tensor: face})

      
        mask_prob = prediction[0, :, 0]  
        no_mask_prob = prediction[0, :, 1]  
     
        if np.mean(mask_prob) > np.mean(no_mask_prob):
            label = "Mask"
            color = (0, 255, 0)
        else:
            label = "No Mask"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return frame

def process_webcam(graph, face_cascade, input_tensor, output_tensor):
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_webcam.avi', fourcc, 20.0, (640, 480))

    with tf.compat.v1.Session(graph=graph) as sess:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = detect_mask(sess, graph, frame, face_cascade, input_tensor, output_tensor)
            out.write(frame)
            cv2.imshow('Face Mask Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def process_video(graph, face_cascade, input_tensor, output_tensor, video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    with tf.compat.v1.Session(graph=graph) as sess:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = detect_mask(sess, graph, frame, face_cascade, input_tensor, output_tensor)
            out.write(frame)
            cv2.imshow('Face Mask Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def process_image(graph, face_cascade, input_tensor, output_tensor, image_path):
    frame = cv2.imread(image_path)
    with tf.compat.v1.Session(graph=graph) as sess:
        frame = detect_mask(sess, graph, frame, face_cascade, input_tensor, output_tensor)
    cv2.imwrite('output_image.jpg', frame)
    cv2.imshow('Face Mask Detection', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = "face_mask_detection.pb" 
    graph = load_model(model_path)

    input_tensor = graph.get_tensor_by_name("data_1:0") 
    output_tensor = graph.get_tensor_by_name("cls_branch_concat_1/concat:0")  

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Uncomment the process you want to run
    # process_webcam(graph, face_cascade, input_tensor, output_tensor)
    process_video(graph, face_cascade, input_tensor, output_tensor, "test.mp4")
    # process_image(graph, face_cascade, input_tensor, output_tensor, "path_to_image.jpg")
    

