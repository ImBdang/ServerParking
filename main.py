from ultralytics import solutions
import ParkingHandle
import cv2
import os
from tqdm import tqdm



def run(model_name, target):
    MODEL_DIR = f"models/{model_name}.pt"

    cap = cv2.VideoCapture(f"{target}.jpg")
    assert cap.isOpened(), "Error reading video file"

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(f"parking_{model_name}.jpg", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    vid_name = f"parking_{model_name}.avi"

    parkingmanager = ParkingHandle.ParkingManagement(        
        model=MODEL_DIR,
        classes=[2]
        # json_file=f"bounding_boxes.json"
    )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=total_frames, desc="Processing Video Frames") as pbar:
        while cap.isOpened():
            ret, im0 = cap.read()
            if not ret:
                pbar.update(total_frames - pbar.n)
                break
            hsv_im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_im0)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced_v = clahe.apply(v)
            enhanced_hsv_im0 = cv2.merge([h, s, enhanced_v])
            im0_processed = cv2.cvtColor(enhanced_hsv_im0, cv2.COLOR_HSV2BGR)
       
            results = parkingmanager(im0_processed, "cam1.json")
            resized_img = cv2.resize(results.plot_im, (800, 600)) 
            cv2.imshow("Parking Detection", resized_img)
         
            video_writer.write(results.plot_im)  
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            pbar.update(1) 

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()  
    os.system(f"mv {vid_name} predict_videos/")

    print("DA CHAY XONG")

def mark_point():
    solutions.ParkingPtsSelection()



# mark_point()

run("yolo12m", "cam1")