import socket

class HttpServer(object):
    def __init__(self):
        tcp_server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        tcp_server_socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        tcp_server_socket.bind(("",8080))
        tcp_server_socket.listen(128)
        self.tcp_server_socket = tcp_server_socket

    def start(self):
        while True:
            new_client_socket,client_ip_port = self.tcp_server_socket.accept()
            print("新客户端来了%s" % (str(client_ip_port)))
            self.request_handler(new_client_socket,client_ip_port)

    def request_handler(self,new_client_socket,ip_port):
        request_data = new_client_socket.recv(1024)
        print(request_data)

        # 判断协议是否为空
        if not request_data:
            print("%s客户端已经下线!"%  str(ip_port))
            new_client_socket.close()

        request_text = request_data.decode()

        # 找到第一个\r\n的位置
        loc = request_text.find("\r\n")

        request_line = request_text[:loc]

        request_line_list = request_line.split(" ")

        #得到请求的资源路径
        file_path = request_line_list[1]
        print("[%s]正在请求:%s!" % (str(ip_port),file_path))

        if file_path == "/":
            file_path = "/index.html"

        response_line = "HTTP/1.1 200 OK\r\n"

        response_header = "Server:Python20WS/2.1\r\n"

        response_blank = "\r\n"

        try:
            with open("static"+file_path,"rb") as file:
                response_body = file.read()
        except Exception as e:
                response_line = "HTTP/1.1 404 Not Found\r\n"
                response_body = "Error! (%s)" % str(e)
                response_body = response_body.encode()

        response_data = (response_line + response_header + response_blank).encode() + response_body

        new_client_socket.send(response_data)
        new_client_socket.close()

if __name__ == "__main__":
    server = HttpServer()
    server.start()