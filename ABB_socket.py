from socket import *

tcpSerSock = socket(AF_INET, SOCK_STREAM)
tcpSerSock.bind(('192.168.137.1',8080))
tcpSerSock.listen(5)

while True:
    print('waiting for connection...')
    tcpCliSock, addr = tcpSerSock.accept()
    print('...connnecting from:', addr)
    data_recv = tcpCliSock.recv(1024)
    print(data_recv.decode())

    while True:
        message = input('>')
        tcpCliSock.send(message.encode('utf-8'))
