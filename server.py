import cv2
import socket
import struct
import pickle
from ultralytics import YOLO

# Load YOLO model
print("Loading YOLOv8n (nano) model...")
model = YOLO("yolov8n.pt")
print("YOLO model loaded successfully.")

# Configure server
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 8000
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(5)
print(f"Server listening on {HOST}:{PORT}")

conn, addr = server_socket.accept()
print(f"Connection established with {addr}")

data = b""
payload_size = struct.calcsize("Q")

try:
    while True:
        # Receive frame from client
        while len(data) < payload_size:
            packet = conn.recv(4096)
            if not packet:
                print("Client disconnected.")
                break
            data += packet

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            data += conn.recv(4096)

        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = pickle.loads(frame_data)

        # Run object detection
        print("Frame received. Running object detection...")
        results = model(frame)
        annotated_frame = results[0].plot()
        print("Object detection completed and frame annotated.")

        # Serialize and send annotated frame back to client
        serialized_frame = pickle.dumps(annotated_frame)
        message = struct.pack("Q", len(serialized_frame)) + serialized_frame
        conn.sendall(message)
        print("Annotated frame sent to client.")
except Exception as e:
    print(f"Server Error: {e}")
finally:
    conn.close()
    server_socket.close()
    print("Server closed.")
