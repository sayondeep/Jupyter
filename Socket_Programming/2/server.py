import socket

HEADERSIZE= 10
# create the socket
# AF_INET == ipv4
# SOCK_STREAM == TCP
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), 1246))
s.listen(5)

while True:
    # now our endpoint knows about the OTHER endpoint.
    clientsocket, address = s.accept()
    print(f"Connection from {address} has been established.")
    msg = 'Hey There!!!'
    msg = f'{len(msg):<{HEADERSIZE}}' + msg
    clientsocket.send(bytes(msg,"utf-8"))

