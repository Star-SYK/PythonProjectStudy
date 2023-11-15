#!/usr/bin/python3
# 文件名：client.py

# 导入 socket、sys 模块
import socket
import sys


def udp_Single():
    # 创建 socket 对象
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 获取本地主机名
    # host = socket.gethostname()
    host = "172.25.128.1"

    # 设置端口号
    port = 8089

    #绑定端口
    udp_socket.bind((host,port))

    # udp_socket.sendto("hello你好".encode('GBK'),("172.25.128.1",8080))

    # 接收小于 1024 字节的数据
    rec_data = udp_socket.recvfrom(1024)
    msg = rec_data[0].decode(encoding="GBK",errors="ignore")
    server_Info = rec_data[1]

    print("接收到",server_Info,"'的消息: ",msg)
    udp_socket.close()

def udp_broadcast():
    while True:
        # 创建 socket 对象
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # 套接字默认不允许发送广播
        udp_socket.setsockopt(socket.SOL_SOCKET,socket.SO_BROADCAST,True)
        udp_socket.sendto("hello你好".encode('GBK'),("172.25.128.1",8080))

        udp_socket.close()

if __name__ == '__main__':
    udp_broadcast()