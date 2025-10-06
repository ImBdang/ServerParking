import requests

url = "http://0.0.0.0:8000/cam1"
url1 = "http://44.203.185.92:8000/cam1"
files = {'file': open('frame1.jpg', 'rb')}
response = requests.post(url1, files=files)
# res = requests.get(url1)

# print(res.json())

print(response.json())




# import cv2, requests

# cap = cv2.VideoCapture(0)
# ret, frame = cap.read()
# cap.release()

# _, buffer = cv2.imencode('.jpg', frame)
# files = {'file': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}

# r = requests.post("http://<server_ip>:<port>/cam1", files=files)
# print(r.json())
