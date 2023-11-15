import socket
import threading


def send_msg(socket):
    """
    发送信息
    :return:
    """
    ipaddress = input("请输入要发送的Ip地址:\n")
    if len(ipaddress) == 0:
        ipaddress = "172.25.128.1"

    port = input("请输入要发送的端口号:\n")
    if len(port) == 0:
        port = "8080"

    message = input("请输入要发送的内容:\n")
    socket.sendto(message.encode('GBK'),(ipaddress,int(port)))

def recv_msg(udp_socket):
    """
    接收信息
    :return:
    """
    # 接收小于 1024 字节的数据
    while True:
        rec_data,ip_port = udp_socket.recvfrom(1024)
        rec_text = rec_data.decode()
        print("接收到", ip_port, "'的消息: ", rec_text)


def main():
    """
   主程序
    :return:
    """
    udp_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

    udp_socket.bind(("",8080))

    thread_recv = threading.Thread(target=recv_msg, args=(udp_socket, ))
    thread_recv.daemon = True
    thread_recv.start()

    while True:
        print("*"*23)
        print("*"*5, " 1、发送消息 ", "*"*5)
        print("*"*5, " 2、退出系统 ", "*"*5)

        #接收用户输入的选项
        sel_num = int(input("请输入选项:\n"))

        match sel_num:
            case 1:
                    print("您选择的是发送消息")
                    send_msg(udp_socket)
            case 2:
                    print("系统正在退出中...")
                    print("系统退出成功！")
    udp_socket.close()



if __name__ == '__main__':
    main()