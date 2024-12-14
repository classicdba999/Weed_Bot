import cv2
import socket
import struct
import pickle

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Configure socket
SERVER_IP = "192.168.0.103"  # Replace with the IP of the remote machine
PORT = 8000
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SERVER_IP, PORT))
print(f"Connected to server at {SERVER_IP}:{PORT}")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Check if the frame is captured correctly
        if frame is None or frame.size == 0:
            print("Error: Captured frame is empty.")
            continue

        # Serialize the frame
        data = pickle.dumps(frame)
        message = struct.pack("Q", len(data)) + data
        client_socket.sendall(message)
        print("Frame sent to server.")

        # Receive the annotated frame from the server
        data = b""
        payload_size = struct.calcsize("Q")

        try:
            # Receive header
            while len(data) < payload_size:
                packet = client_socket.recv(4096)
                if not packet:
                    raise ConnectionError("Server disconnected.")
                data += packet

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]

            # Receive the complete message
            while len(data) < msg_size:
                packet = client_socket.recv(4096)
                if not packet:
                    raise ConnectionError("Server disconnected.")
                data += packet

            annotated_frame_data = data[:msg_size]
            annotated_frame = pickle.loads(annotated_frame_data)

            # Display the annotated frame
            cv2.imshow("Annotated Video Feed", annotated_frame)

            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q' key pressed. Exiting frame capture loop...")
                break
        except ConnectionError as e:
            print(f"Connection error: {e}")
            break
        except struct.error as e:
            print(f"Data error: {e}")
            break
except KeyboardInterrupt:
    print("Interrupted. Closing connection.")
finally:
    cap.release()
    client_socket.close()
    cv2.destroyAllWindows()
    print("Connection closed.")
