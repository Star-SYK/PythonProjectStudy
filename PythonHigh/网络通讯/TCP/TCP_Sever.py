import socket

while True:
    tcp_server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

    tcp_server.bind(("172.25.128.1",8080))

    tcp_server.listen(128)

    new_client_socket ,client_ip_port= tcp_server.accept()
    print("新客户端来了%s" % (str(client_ip_port)))

    while True:
        recv_data = new_client_socket.recv(1024)
        if recv_data:
            recv_text = recv_data.decode("GBK")
            print("接收到[%s]的信息：%s" % (str(client_ip_port), recv_text))
        else:
            print("客户端%s已断开" % (str(client_ip_port)))
            break
    new_client_socket.close()
    tcp_server.close()