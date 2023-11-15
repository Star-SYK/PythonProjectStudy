import socket

tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

tcp_client.connect(("172.25.128.1", 8080))

tcp_client.send("你好TCP_Server".encode("GBK"))

recv_data =  tcp_client.recvfrom(1024)

print(recv_data[0].decode("GBK"))

tcp_client.close()