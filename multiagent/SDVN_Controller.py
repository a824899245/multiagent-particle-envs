# import mcds
import math
import networkx as nx
import numpy as np
from .Global_Par import *
from .Packet import *
from .dij_test1 import *
import time as tim
from sklearn.cluster import KMeans, MiniBatchKMeans

rect_list = []
draw_list = []
time = 0


def rout(v_c, v_c_d, hop, num, vehicle_position, next_hop):
    if hop.count(0) == 0 or num > 5:
        return
    for i in range(len(hop)):
        if hop[i] == 0:
            min = 99999
            for j in range(len(vehicle_position)):
                if hop[j] == num:
                    dis = math.sqrt(pow(vehicle_position[i][0] - vehicle_position[j][0], 2) + pow(
                        vehicle_position[i][1] - vehicle_position[j][1], 2))
                    dis = float("%.5f" % dis)
                    if dis < com_dis:
                        if dis + v_c_d[j] < min:
                            min = dis + v_c_d[j]
                            mini = j
            if min != 99999:
                hop[i] = num
                v_c_d[i] = min
                v_c[i] = v_c[mini]
                next_hop[i] = j
    rout(v_c, v_c_d, hop, num + 1, vehicle_position, next_hop)


def isOverlap(minx0, miny0, maxx0, maxy0, minx1, miny1, maxx1, maxy1):
    if (maxx0 > minx1) and (maxx1 > minx0) and (maxy0 > miny1) and (maxy1 > miny0):
        return 1
    else:
        return 0


class Stack(object):
    """栈"""

    def __init__(self):
        self.items = []

    def is_empty(self):
        """判断是否为空"""
        return self.items == []

    def push(self, item):
        """加入元素"""
        self.items.append(item)

    def pop(self):
        """弹出元素"""
        return self.items.pop()

    def peek(self):
        """返回栈顶元素"""
        return self.items[len(self.items) - 1]

    def size(self):
        """返回栈的大小"""
        return len(self.items)


def getstack(adj, i, n, s):
    for j in range(i):
        if adj[i][j] == 1:
            s.push(j)
    return s.size()


class local_Controller:
    def __init__(self, position, no):
        self.no = no
        self.position = position
        self.switch = 0
        self.sur_vehicle = []


class c_position:
    def __init__(self, x, y):
        self.postion = [x, y]
        self.edge = []


class SDVNController:
    def __init__(self, junction_matrix, node_num):
        self.total_controller_list = []
        self.total_fixed_controller_list = []
        self.hello_list = []  # hello请求列表
        self.flow_request_list = []  # 路由请求列表
        self.geo_flow_request_list = []
        self.flow_error_list = []  # 错误请求列表
        self.junction_matrix = junction_matrix  # 邻接矩阵
        self.node_info_dict = {i: [[], [], [], ] for i in range(node_num)}  # 所有节点信息
        self.v_c = []
        self.v_c_d = []
        self.controller_position = []

    # 根据hello列表中的条目更新控制器中的节点信息
    def predict_position(self):
        for value in self.hello_list:
            self.node_info_dict[value.node_id] = [value.position, value.velocity, value.acceleration,
                                                  value.current_cache]
        self.hello_list.clear()
        return

    def junction_matrix_construction(self, node_num):
        self.junction_matrix.clear()
        for i in range(0, node_num):
            for j in range(0, i):
                a = pow(self.node_info_dict[i][0][0] - self.node_info_dict[j][0][0], 2) + pow(
                    self.node_info_dict[i][0][1] - self.node_info_dict[j][0][1], 2)
                if a < pow(com_dis, 2):
                    self.junction_matrix.add_edge(i, j, weight=a)
                    self.junction_matrix.add_edge(j, i, weight=a)

    # 根据节点信息计算路由
    def calculate_path(self, x_id, des_id, node_list, node_num):
        # bellman-ford
        # route = bf.bellman_ford(self.junction_matrix, x_id, des_id, self.junction_matrix.number_of_nodes())
        # if route:
        #     print(route)
        #     return route
        # print('%d to %d calculation error' % (x_id, des_id))
        # return [x_id, des_id]

        # dijkstra
        try:
            Dijkstra(self.junction_matrix, x_id, des_id)
        except nx.NetworkXError as err1:
            route = None
        else:
            route = Dijkstra(self.junction_matrix, x_id, des_id)

        if route:
            print(route)
            with open('dijkstra.txt', 'a') as f:
                a = ""
                for i in route:
                    a += str(i)
                    a += ' '
                a += '\n'
                f.write(a)
            return route
        print('%d to %d calculation error' % (x_id, des_id))
        return [x_id, des_id]

        # 自己的算法
        # reward = [[0 for i in range(80)] for i in range(80)]
        # jh.junction_reward(reward, node_list[des_id].junction[0])
        # h_s1, h_s2 = jh.hidden_seq_generate(reward, node_list[x_id].junction[0], node_list[des_id].junction[0])
        # ji.e_arrival_time[x_id] = 0
        # jh.hidden_to_obverse(x_id, des_id, node_list, h_s1)
        # jh.hidden_to_obverse(x_id, des_id, node_list, h_s2)
        # # jh.hidden_to_obverse_1(x_id, des_id, node_list, h_s1, h_s2)
        # a, b = tg.earliest_arrival(ji.edge_list, x_id, des_id, node_num)
        # route = []
        # tg.s_routing(b, x_id, des_id, route)
        #
        # if route:
        #     print(route)
        #     return route
        # print('%d to %d calculation error' % (x_id, des_id))
        # return [x_id, des_id]
        #
        # # 自己的算法
        # reward = [[0 for i in range(80)] for i in range(80)]
        # jh.junction_reward(reward, node_list[des_id].junction[0])
        # h_s1, h_s2 = jh.hidden_seq_generate(reward, node_list[x_id].junction[0], node_list[des_id].junction[0])
        # ji.e_arrival_time[x_id] = 0
        # jh.hidden_to_obverse(x_id, des_id, node_list, h_s1)
        # jh.hidden_to_obverse(x_id, des_id, node_list, h_s2)
        # # jh.hidden_to_obverse_1(x_id, des_id, node_list, h_s1, h_s2)
        # a, b = tg.earliest_arrival(ji.edge_list, x_id, des_id, node_num)
        # route = []
        # tg.s_routing(b, x_id, des_id, route)
        #
        # if route:
        #     print("vehicle")
        #     print(route)
        #     re_list = []
        #     for i in route:
        #         re_list.append([self.node_info_dict[i][0][0], self.node_info_dict[i][0][1]])
        #     data = np.array(re_list)
        #     fig = plt.figure(figsize=(8, 7.82), dpi=80)
        #     plt.plot(data[:, 0], data[:, 1], color='r')
        #     plt.scatter(data[:, 0], data[:, 1], edgecolors='black')
        #     plt.show()
        #     return route
        # route = dij.Dijkstra(self.junction_matrix, x_id, des_id)
        # if route:
        #     print("vehicle")
        #     print(route)
        #     return route
        # print('%d to %d calculation error' % (x_id, des_id))
        # return [x_id, des_id]

        # # # 自己的算法_big
        # reward = [[0 for i in range(268)] for i in range(268)]
        # bjh.junction_reward(reward, node_list[des_id].big_junction)
        # h_s1, h_s2 = bjh.hidden_seq_generate(reward, node_list[x_id].big_junction, node_list[des_id].big_junction)
        # bji.e_arrival_time[x_id] = 0
        # bjh.hidden_to_obverse(x_id, des_id, node_list, h_s1)
        # bjh.hidden_to_obverse(x_id, des_id, node_list, h_s2)
        # # jh.hidden_to_obverse_1(x_id, des_id, node_list, h_s1, h_s2)
        # a, b = tg.earliest_arrival(ji.edge_list, x_id, des_id, node_num)
        # route = []
        # tg.s_routing(b, x_id, des_id, route)
        #
        # if route:
        #     print("vehicle")
        #     print(route)
        #     re_list = []
        #     for i in route:
        #         re_list.append([self.node_info_dict[i][0][0], self.node_info_dict[i][0][1]])
        #     data = np.array(re_list)
        #     fig = plt.figure(figsize=(8, 7.82), dpi=80)
        #     plt.plot(data[:, 0], data[:, 1], color='r')
        #     plt.scatter(data[:, 0], data[:, 1], edgecolors='black')
        #     plt.show()
        #     return route
        # try:
        #     nx.shortest_path(self.junction_matrix, source=x_id, target=des_id)
        # except nx.NodeNotFound as err1:
        #     route = None
        # except nx.NetworkXNoPath as err2:
        #     route = None
        # else:
        #     route = nx.shortest_path(self.junction_matrix, source=x_id, target=des_id)
        # if route:
        #     print("vehicle")
        #     print(route)
        #     return route
        # print('%d to %d calculation error' % (x_id, des_id))
        # return [x_id, des_id]
        # #
        # # HRLB
        # route = hr.routing(x_id, des_id, node_list)
        # if len(route) > 2:
        #     with open('HRLB.txt', 'a') as f:
        #         a = ""
        #         for i in route:
        #             a += str(i)
        #             a += ' '
        #         a += '\n'
        #         f.write(a)
        #     print(route)
        #     return route
        # else:
        #     try:
        #         dij.Dijkstra(self.junction_matrix, x_id, des_id)
        #     except nx.NetworkXError as err1:
        #         route = None
        #     else:
        #         route = dij.Dijkstra(self.junction_matrix, x_id, des_id)
        #
        #     if route:
        #         print(route)
        #         with open('HRLB.txt', 'a') as f:
        #             a = ""
        #             for i in route:
        #                 a += str(i)
        #                 a += ' '
        #             a += '\n'
        #             f.write(a)
        #         return route
        #     print('%d to %d calculation error' % (x_id, des_id))
        #     return [x_id, des_id]

        # # HRLB_big
        # route = bhr.routing(x_id, des_id, node_list)
        # if len(route) > 2:
        #     print(route)
        #     return route
        # else:
        #     print("error")
        #     # dijkstra
        #     try:
        #         dij.Dijkstra(self.junction_matrix, x_id, des_id)
        #     except nx.NetworkXError as err1:
        #         route = None
        #     else:
        #         route = dij.Dijkstra(self.junction_matrix, x_id, des_id)

        #     if route:
        #         print(route)
        #         return route
        #     print('%d to %d calculation error' % (x_id, des_id))
        #     return [x_id, des_id]

        # # hmmm
        # hm.inti()
        # route = hm.routing(x_id, des_id, node_list, 10)
        # if route and len(route) != 2:
        #     print(route)
        #     with open('PRHMM.txt', 'a') as f:
        #         a = ""
        #         for i in route:
        #             a += str(i)
        #             a += ' '
        #         a += '\n'
        #         f.write(a)
        #     return route
        # else:
        #     try:
        #         nx.shortest_path(self.junction_matrix, source=x_id, target=des_id)
        #     except nx.NodeNotFound as err1:
        #         route = None
        #     except nx.NetworkXNoPath as err2:
        #         route = None
        #     else:
        #         route = nx.shortest_path(self.junction_matrix, source=x_id, target=des_id)
        #     if route:
        #         print(route)
        #         with open('PRHMM.txt', 'a') as f:
        #             a = ""
        #             for i in route:
        #                 a += str(i)
        #                 a += ' '
        #             a += '\n'
        #             f.write(a)
        #         return route
        #     print('%d to %d calculation error' % (x_id, des_id))
        #     return [x_id, des_id]

    def calculate_path_1(self, x_id, des_id, node_list, node_num, neighbour):
        # bellman-ford
        # route = bf.bellman_ford(self.junction_matrix, x_id, des_id, self.junction_matrix.number_of_nodes())
        # if route:
        #     print(route)
        #     return route
        # print('%d to %d calculation error' % (x_id, des_id))
        # return [x_id, des_id]
        # dijkstra
        self.junction_matrix
        try:
            Dijkstra(self.junction_matrix, neighbour, des_id)
        except nx.NetworkXError as err1:
            route = None
        else:
            route = Dijkstra(self.junction_matrix, neighbour, des_id)

        if route:
            print(route)
            with open('dijkstra.txt', 'a') as f:
                a = ""
                for i in route:
                    a += str(i)
                    a += ' '
                a += '\n'
                f.write(a)
            route1 = [x_id]
            route1.extend(route)
            return route1
        print('%d to %d calculation error' % (x_id, des_id))
        return [x_id, neighbour, des_id]

    @staticmethod
    def geo_calculate_path(x_id, des_list, node_list):
        # sub = des_list
        # sub.append(x_id)
        # G = nx.Graph()
        # for a in sub:
        #     for b in sub:
        #         if a == 381:
        #             a = a-1
        #         if b == 381:
        #             b = b-1
        #         if a != b:
        #             d = pow(node_list[a].position[0] - node_list[b].position[0], 2) + pow(
        #                 node_list[a].position[1] - node_list[b].position[1], 2)
        #             G.add_edge(a, b, weight=d)
        # un = des_list
        # visited = [x_id]
        # next_hop = [[] for i in range(len(node_list))]
        # mcds.dfs(G, x_id, visited, un, next_hop)
        # return visited, next_hop
        return 1, 1

    # 向路由上的每个节点发送路由回复
    @staticmethod
    def send_reply(x_id, des_id, route, node_list, node_id, seq, dur):
        print("duration=")
        print(dur)
        flow_reply = FlowReply(x_id, des_id, route, node_id, seq, dur)
        for node_num in route:
            node_list[node_num].receive_flow(flow_reply)
        # 时延处理
        return

    @staticmethod
    def geo_send_reply(x_id, des_list, associated_node, next_hop_list, node_list, node_id, seq):
        for node in associated_node:
            flow_reply = geo_FlowReply(x_id, des_list, next_hop_list[node], node_id, seq)
            node_list[node].geo_receive_flow(flow_reply)
        # 时延处理
        return

    # 处理请求表中的每个请求，计算路由，发送回复
    def resolve_request(self, node_list):
        for request in self.flow_request_list:
            t = request.time
            route = self.calculate_path(request.source_id, request.des_id, node_list, len(node_list))
            dur = tim.time() - t
            self.send_reply(request.source_id, request.des_id, route, node_list, request.node_id, request.seq, dur)
        self.flow_request_list.clear()
        return route

    # 处理请求表中的每个请求，计算路由，发送回复
    def resolve_request_1(self, node_list, neighbour):
        for request in self.flow_request_list:
            t = request.time
            route = self.calculate_path_1(request.source_id, request.des_id, node_list, len(node_list), neighbour)
            dur = tim.time() - t
            self.send_reply(request.source_id, request.des_id, route, node_list, request.node_id, request.seq, dur)
        self.flow_request_list.clear()
        return

    def geo_resolve_request(self, node_list):
        for request in self.geo_flow_request_list:
            associated_node, next_hop_list = self.geo_calculate_path(request.source_id, request.des_list, node_list)
            self.geo_send_reply(request.source_id, request.des_list, associated_node, next_hop_list, node_list,
                                request.node_id, request.seq)
        self.geo_flow_request_list.clear()
        return

    # 删除路由信息（超过三次需要删除所有相关路由信息与分组）
    def delete_routing_pkt(self, node_list, source_id, id, seq, des_id):
        # 到达目的节点后，删除相关信息并返回
        if id == des_id:
            for table in node_list[id].routing_table[::-1]:
                if table.seq == seq and table.node_id == source_id:
                    # print('node %d routing delete' % id)
                    node_list[id].routing_table.remove(table)
            for pkt in node_list[id].data_pkt_list[::-1]:
                if pkt.seq == seq and pkt.node_id == source_id:
                    # print('node %d pkt delete' % id)
                    node_list[id].data_pkt_list.remove(pkt)
            return
        # 未到达目的节点，根据路由表递归地删除。
        for table in node_list[id].routing_table[::-1]:
            if table.seq == seq and table.node_id == source_id:
                self.delete_routing_pkt(node_list, source_id, table.next_hop_id, seq, des_id)
                # print('node %d routing delete' % id)
                node_list[id].routing_table.remove(table)
        for pkt in node_list[id].data_pkt_list[::-1]:
            if pkt.seq == seq and pkt.node_id == source_id:
                # print('node %d pkt delete' % id)
                node_list[id].data_pkt_list.remove(pkt)

    # 解析错误请求信息
    def resolve_error(self, node_list):
        # 对错误请求列表里的所有节点处理
        for error in self.flow_error_list[::-1]:
            # 同一跳错误次数大于N次，此条路由失败
            if error.time > re_time:
                # print('%3d to %3d 路由失败 %3d %3d' % (error.error_id, error.des_id, error.source_id, error.source_seq))
                # 删除相关路由
                self.delete_routing_pkt(node_list, error.source_id, error.error_id, error.source_seq, error.des_id)
                fail_time = fail_time + 1
                # print('source %d seq %d des %d err %d' % (error.source_id, error.source_seq, error.des_id, error.error_id))
                # print('delete\n')
                self.flow_error_list.remove(error)
        # 不然计算路由， 向下下发
        for error1 in self.flow_error_list:
            error1.time += 1
            route = self.calculate_path(error1.error_id, error1.des_id, node_list, len(node_list))
            dur = tim.time() - error1.s_time
            self.send_reply(error1.error_id, error1.des_id, route, node_list, error1.source_id, error1.source_seq, 0.3)
        return

    def initial_placement(self, intersection_position, intersection_matrix, model):
        delta_x = 6440/8
        delta_y = 6289/8
        x = delta_x
        y = delta_y
        for i in range(4):
            for j in range(4):
                a = [x+delta_x*2*i,y+delta_y*2*j,100*model]
                self.total_controller_list.append(a)
        # ## 调用控制器布置算法，获得调整控制器位置和固定控制器位置
        # position_list,position_list_fixed = cover.initial_placement(intersection_position,intersection_matrix)
        # ## 返回所有控制器位置，算法完成 (＾o＾)ﾉ ​​​​
        # ## 将其分别录入控制器属性中
        # a = np.array(position_list)
        # b = np.array(position_list_fixed)
        # np.savetxt("totalcl.txt",np.delete(a, [2,3], axis=1))
        # np.savetxt("totalclf.txt",np.delete(b, [2,3], axis=1))
        # for i in position_list:
        #     self.total_controller_list.append([i[0],i[1]])
        # for i in position_list_fixed:
        #     self.total_fixed_controller_list.append([i[0],i[1]])

        # a = np.loadtxt("totalcl.txt")
        # b = np.loadtxt("totalclf.txt")
        # c = np.array([np.insert(i, 2, np.array([0]), axis=0) for i in a])
        # d = np.array([np.insert(i, 2, np.array([0]), axis=0) for i in b])
        # self.total_controller_list = c.tolist()
        # self.total_fixed_controller_list = d.tolist()

    def controller_place(self,model,k):
        k = int(k)
        on_controller = []
        if(model < 2 ):
            delta_x = 3350.24 / (k*2)
            delta_y = 2860.33  / (k*2)
            x = delta_x
            y = delta_y
            for i in range(k):
                for j in range(k):
                    a = [x + delta_x * 2 * i, y + delta_y * 2 * j, 100 * model]
                    on_controller.append(a)
        elif(model == 2):
            delta_x = 6440 / (2 * 2)
            delta_y = 6289 / (2 * 2)
            x = delta_x
            y = delta_y
            for i in range(2):
                for j in range(2):
                    a = [x + delta_x * 2 * i, y + delta_y * 2 * j, 0]
                    on_controller.append(a)

            vehicle = []
            for i in range(len(self.node_info_dict)):
                vehicle.append([self.node_info_dict[i][0][0],self.node_info_dict[i][0][1]])
            data = np.array(vehicle)
            km = MiniBatchKMeans(init='k-means++', n_clusters= 5 , batch_size=100, random_state=28)
            km.fit(data)
            centroids = km.cluster_centers_
            # centroids = kmeanspp.get_centroids(data, k ** 2)
            while (centroids.min() == float('nan')):
                km = MiniBatchKMeans(init='k-means++', n_clusters= 5 , batch_size=100, random_state=28)
                km.fit(data)
                centroids = km.cluster_centers_
            for i in range(k**2):
                on_controller.append([centroids[i,0],centroids[i,1],100])
        elif(model == 3):
            vehicle = []
            for i in range(len(self.node_info_dict)):
                vehicle.append([self.node_info_dict[i][0][0], self.node_info_dict[i][0][1]])
            data = np.array(vehicle)
            km = KMeans(n_clusters=k**2,init='k-means++',n_init=10,max_iter=500,tol=0.0001)
            km.fit(data)
            centroids = km.cluster_centers_
            # centroids = kmeanspp.get_centroids(data, k ** 2)
            while (centroids.min() == float('nan')):
                km = KMeans(n_clusters=k ** 2, init='k-means++', n_init=10, max_iter=500, tol=0.0001)
                km.fit(data)
                centroids = km.cluster_centers_
            for i in range(k**2):
                on_controller.append([centroids[i,0],centroids[i,1],100])
        # on_controller = []
        # for i in self.total_controller_list:
        #     on_controller.append(i)
        # for i in self.total_fixed_controller_list:
        #     on_controller.append(i)

        # for i in range(len(self.node_info_dict)):
        #     vehicle.append([self.node_info_dict[i][0][0],self.node_info_dict[i][0][1]])
        # print("所有控制器数")
        # print(len(self.total_controller_list)+len(self.total_fixed_controller_list))
        # sol = ad.multiobj(len(self.total_controller_list),vehicle,self.total_controller_list,self.total_fixed_controller_list)
        # s = np.array(sol)
        # s = s.sum(axis=1)
        # mi = np.argmax(s)
        # sol[1] = [1 for i in range(len(sol[1]))]
        # ## 获取现在所有开启控制器位置信息
        # for i in self.total_fixed_controller_list:
        #     on_controller.append(i)
        # for i in range(len(sol[mi])-2):
        #     if sol[mi][i] == 1:
        #         on_controller.append(self.total_controller_list[i])
        # print("开启控制器数")
        # print(s[mi]+len(self.total_fixed_controller_list))
        # 位置数据处理

        # a = np.array(vehicle)
        # fig = plt.figure(figsize=(8, 7.82), dpi=80)
        # plt.scatter(a[:, 0], a[:, 1], alpha=0.5, edgecolors='blue')
        # plt.scatter(a[:, 0], a[:, 1], alpha=1, s=1, edgecolors='black')
        # a = []
        #
        # a = np.array(on_controller)
        # plt.scatter(a[:, 0], a[:, 1], alpha=0.5, s=Gp.con_dis * Gp.con_dis * 0.024, c='y', edgecolors='blue')
        # plt.scatter(a[:, 0], a[:, 1], alpha=1, s=1, edgecolors='black')
        # plt.show()

        ## 将其录入控制器属性中
        self.controller_position = on_controller

    def vehicle_controller(self):
        v_c_d = []
        v_c = []
        ## 计算车辆到控制器间距离，计算每辆车离自己最近的控制器与对应距离
        vehicle_position = []
        for i in range(len(self.node_info_dict)):
            vehicle_position.append([self.node_info_dict[i][0][0], self.node_info_dict[i][0][1], 0.0])
        on_controller = self.controller_position
        for i in range(len(vehicle_position)):
            min = 999999
            for j in range(len(on_controller)):
                d = np.linalg.norm(np.array(vehicle_position[i]) - np.array(on_controller[j]))
                # d = float("%.5f" % d)
                if d < min:
                    min = d
                    minj = j
            v_c_d.append(min)
            v_c.append(minj)
        self.v_c_d = np.array(v_c_d)
        self.v_c = np.array(v_c)

        # xa = len(vehicle_position)
        # xb = len(on_controller)
        # # 联系每一个解和对应的向量，即n辆车到m个控制器的位置，a为车辆位置矩阵，b为控制器位置矩阵
        # aq = np.array(vehicle_position)
        # bq = np.array(on_controller)
        # ## 计算asq
        # a = aq ** 2
        # a = a.sum(axis=1)
        # aaa = a
        # for i in range(xb - 1):
        #     aaa = np.vstack([aaa, a])
        # aaa = aaa.transpose()
        # ## 计算bsq
        # b = bq ** 2
        # b = b.sum(axis=1)
        # b = b.transpose()
        # bbb = b
        # for i in range(xa - 1):
        #     bbb = np.vstack([bbb, b])
        # ## 计算 a*(b^-1)
        # c = aq.dot(bq.transpose())
        # ## asq+bsq-2*a*(b^-1)开方即为n到m的距离矩阵
        # ddd = np.sqrt(aaa + bbb - 2 * c)
        # ## 在每行中挑出最小的，即为车辆的最距离控制器距离，索引通过argmin获得
        # self.v_c_d = np.min(ddd.T, 0)
        # self.v_c = np.argmin(ddd.T, 0)

        ## 新算法(＾o＾)ﾉ♡(＾o＾)ﾉ♡(＾o＾)ﾉ♡(＾o＾)ﾉ♡(＾o＾)ﾉ♡
        # hop统计到最近控制器的跳数
        hop = [0 for i in range(len(self.v_c_d))]
        # next_hop统计到最近控制器的下一跳
        next_hop = [0 for i in range(len(self.v_c_d))]
        for i in range(len(self.v_c_d)):
            if self.v_c_d[i] <= com_dis:
                hop[i] = 1
        for k in range(1, 4):
            ## 如果hop为0，说明一跳够不着
            for i in range(len(hop)):
                if hop[i] == 0:
                    min = 99999
                    ## 找自己车辆最大通信范围内的所有车辆，如果有车此时hop标识为1，说明可以通过该车辆传输，比较距离，选最小的
                    for j in range(len(vehicle_position)):
                        if hop[j] == k:
                            d = np.linalg.norm(np.array(vehicle_position[i]) - np.array(vehicle_position[j]))
                            # d = float("%.5f" % d)
                            if d < com_dis:
                                if d + self.v_c_d[j] < min:
                                    min = d + self.v_c_d[j]
                                    mini = j
                    ## 更新跳数为2，更新最近控制器的距离，为hop为1的车辆到控制器的距离+自己车辆到hop为1车辆的距离
                    ## 更新下一跳节点为hop为1车辆，更新最近控制器与hop为1车辆的最近控制器保持一致
                    if min != 99999:
                        hop[i] = k + 1
                        self.v_c_d[i] = min
                        self.v_c[i] = self.v_c[mini]
                        next_hop[i] = mini
        ## 如果离最近控制器的距离小于最大通信距离，可传输，hop设为一跳

        # ## 接下来 代码相同，只是考虑跳数为3，为4，为5，逐步更新，直到，跳数》5或者没有车辆未被统计
        # ## 这里应该写成递归，那样比较漂亮，奈何python函数机制没搞懂，先写成这样五块，之后再改进
        # for i in range(len(hop)):
        #     if hop[i] == 0:
        #         min = 99999
        #         for j in range(len(vehicle_position)):
        #             if hop[j] == 2:
        #                 d = np.linalg.norm(np.array(vehicle_position[i]) - np.array(vehicle_position[j]))
        #                 if d < Gp.com_dis:
        #                     if d + self.v_c_d[j] < min:
        #                         min = d + self.v_c_d[j]
        #                         mini = j
        #         if min != 99999:
        #             hop[i] = 3
        #             self.v_c_d[i] = min
        #             self.v_c[i] = self.v_c[mini]
        #             next_hop[i] = mini
        #
        # for i in range(len(hop)):
        #     if hop[i] == 0:
        #         min = 99999
        #         for j in range(len(vehicle_position)):
        #             if hop[j] == 3:
        #                 d = np.linalg.norm(np.array(vehicle_position[i]) - np.array(vehicle_position[j]))
        #                 if d < Gp.com_dis:
        #                     if d + self.v_c_d[j] < min:
        #                         min = d + self.v_c_d[j]
        #                         mini = j
        #         if min != 99999:
        #             hop[i] = 4
        #             self.v_c_d[i] = min
        #             self.v_c[i] = self.v_c[mini]
        #             next_hop[i] = mini
        #
        # for i in range(len(hop)):
        #     if hop[i] == 0:
        #         min = 99999
        #         for j in range(len(vehicle_position)):
        #             if hop[j] == 4:
        #                 d = np.linalg.norm(np.array(vehicle_position[i]) - np.array(vehicle_position[j]))
        #                 if d < Gp.com_dis:
        #                     if d + self.v_c_d[j] < min:
        #                         min = d + self.v_c_d[j]
        #                         mini = j
        #         if min != 99999:
        #             hop[i] = 5
        #             self.v_c_d[i] = min
        #             self.v_c[i] = self.v_c[mini]
        #             next_hop[i] = mini
        gar_veh =[]
        con_veh = [[] for a in range(len(self.controller_position))]
        for i in range(len(v_c)):
            if hop[i] == 0:
                gar_veh.append(vehicle_position[i])
                gar_veh += 1
            else:
                con_veh[v_c[i]].append(vehicle_position[i])

        # color = ['g', 'r', 'c', 'm', 'y', 'k', 'w']
        # cnames = [
        #     '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
        #     '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
        #     '#000075', '#a9a9a9', '#7B68EE', '#4682B4']
        #
        # for i in gar_veh:
        #     plt.plot(i[0], i[1],'xk',markersize=4)
        # for i in range(len(self.controller_position)):
        #     co = cnames[i%18]
        #     plt.plot(self.controller_position[i][0], self.controller_position[i][1], 'o', color =co, markersize=16)
        #     for j in con_veh[i]:
        #         plt.plot(j[0], j[1],'o', color =co, markersize=4)
        # plt.show()



        return hop, next_hop
