import socket
HEADERSIZE= 10
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), 1246))
# msg = s.recv(8)
# print(msg.decode("utf-8"))

# while True:
#     msg = s.recv(8)
#     print(msg.decode("utf-8"))

full_msg = ''
new_msg=True

while True:
    msg = s.recv(16)
    if new_msg:
        print(f'new message length: {msg[:HEADERSIZE]}')
        msglen = int(msg[:HEADERSIZE])
        new_msg = False

    full_msg += msg.decode("utf-8")

    if(len(full_msg)-HEADERSIZE==msglen):
        print('full message recvd')
        print(full_msg[HEADERSIZE:])
        new_msg = True
        full_msg=''

print(full_msg)


