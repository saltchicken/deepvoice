import socket

def send_message(message, host='127.0.0.1', port=9986):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((host, port))
        
        # Send data in chunks
        client_socket.sendall(message.encode() + b"EOF")  # Append EOF marker
        
        # Receive response
        # response = client_socket.recv(4096).decode()
        # print(f"Server responded: {response}")

if __name__ == "__main__":
    try:
        while True:
            user_input = input("Enter your message: ")
            send_message(user_input)
    except KeyboardInterrupt:
        print("Exiting")

