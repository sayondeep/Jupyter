import socket

# create the socket
# AF_INET == ipv4
# SOCK_STREAM == TCP
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), 1245))
s.listen(5)

while True:
    # now our endpoint knows about the OTHER endpoint.
    clientsocket, address = s.accept()
    print(f"Connection from {address} has been established.")
    clientsocket.send(bytes("Hey there!!!","utf-8"))
    clientsocket.close()
