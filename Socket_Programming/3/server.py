import socket
import time
import pickle



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
    d = {1: "Hey", 2: "There"}

    msg = pickle.dumps(d)
    # print(msg)
    msg = bytes(f'{len(msg):<{HEADERSIZE}}',"utf-8")+ msg
    clientsocket.send(msg)