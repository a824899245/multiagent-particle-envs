from __future__ import absolute_import, division, print_function
import numpy as np
from .Get_Move import get_position
from .Init import init_controller, init_node
import random
from .A2G_channel import A2G_channel_1, band
import math

class SDVN_U(object):

    def __init__(self, vehicle_num=200, control_num = 25, channel_num = 7):
        # the map of the deep sea treasure (convex version)
        self.reward_dim = 2
        self.vehicle_num = vehicle_num # zero based depth
        self.control_num = control_num
        self.channel_num = channel_num
        input = "rural_big_"+str(vehicle_num)+".mobility.tcl"
        head = []
        tail = []
        with open(input, 'r') as f:
            for line in f:
                if line[2] == 'o':
                    tail.append(line)
                else:
                    head.append(line)
        with open('tiexi1.tcl', 'w') as f:
            for line in head:
                f.write(line)
            for line in tail:
                f.write(' ')
                f.write(line)

        self.node_list = []
        self.com_node_list = []
        self.sum = 0
        self.flag = 1
        self.time_slot_duration = 500
        # 位置文件读取
        self.movement_matrix, self.init_position_matrix = get_position('tiexi1.tcl')
        # 控制器初始化
        self.controller = init_controller(self.vehicle_num)

        # 位置数据处理
        self.init_position_arranged = self.init_position_matrix[np.lexsort(self.init_position_matrix[:, ::-1].T)]

        self.node_position = self.init_position_arranged[0]
        self.next_node_position = self.init_position_arranged[0]
        self.total_route = []
        self.shop = 0
        self.node_list = (init_node(self.node_position, self.controller))  
        self.start_time = random.randint(0,1100-(100*self.time_slot_duration/1000))
        self.threshold = -10
        self.speed_limit = 20
        current_move = self.movement_matrix[np.nonzero(self.movement_matrix[:, 0].A == self.start_time)[0], :]
        for value in current_move:
            for i in range(1, 4):
                self.node_position[int(value[0, 1]), i] = value[0, i + 1]
        self.node_id_position = self.node_position[:, [1, 2, 3]]

        self.position_state = np.array(self.node_id_position)

        for node in self.node_list:
            node.update_node_position(self.node_id_position)
            node.generate_hello(self.controller)
        # 控制器更新网络全局情况
        self.controller.predict_position()
        ## 如果达到条件，对可调整控制器位置进行多目标ABC，获得开关情况
        self.controller.init_controller_place(2)
        self.control_state = []
        for i in self.controller:
            self.control_state.append(i.controller_position)
        self.control_state = np.array(self.control_state)

        self.channel_penalty = 0
        self.power_penalty = 0
        self.flight_penalty = 0

        self.current_AoI = np.zeros(vehicle_num)
        self.throughput = np.zeros(vehicle_num)
        self.c_v_co_association = np.zeros(vehicle_num,control_num)
        self.c_v_ch_association = np.zeros(vehicle_num,channel_num)
        self.t_v_co_association = np.zeros(vehicle_num,control_num)
        self.t_v_ch_association = np.zeros(vehicle_num,channel_num)
        self.c_v_power = np.zeros(vehicle_num)
        self.t_v_power = np.zeros(vehicle_num)
        self.AoI_sum = 0
        self.throughput_sum = 0
                    # ### 200 * 25

                    # c_v_ch_association = controller.c_vehicle_channel()
                    # ### 200 * 7

                    # t_v_co_association = controller.t_vehicle_controller()
                    # ### 200 * 25

                    # t_v_ch_association = controller.t_vehicle_channel()
                    # ### 200 * 7

                    # c_v_power = controller.c_power()
                    # ### 200 * 1

                    # t_v_power = controller.t_power()

  

        b = np.zeros(vehicle_num)
        b1 = np.ones(control_num)

        self.position_state = np.insert(self.position_state, 3, values=b, axis=1)
        self.control_state = np.insert(self.control_state, 3, values=b1, axis=1)

        self.current_state = np.concatenate((self.position_state,self.control_state))
        self.terminal = False

        # DON'T normalize
        self.max_reward = 10.0

        # state space specification: 2-dimensional discrete box
        self.state_spec = [['discrete', 1, [0, self.vehicle_num+self.control_num]],
                           ['discrete', 1, [0, self.tree_depth]],
                           ['discrete', 1, [0, 2 ** self.tree_depth - 1]]]

        # action space specification: 0 left, 1 right
        self.action_spec = ['discrete', 1, [0, 2]]

        # reward specification: 2-dimensional reward
        # 1st: treasure value || 2nd: time penalty
        self.reward_spec = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
        

    def get_ind(self, pos):
        return int(2 ** pos[0] - 1) + pos[1]

    def get_tree_value(self, pos):
        return self.tree[self.get_ind(pos)]

    def reset(self):
        '''
            reset the location of the submarine
        '''
        self.start_time = random.randint(0,1100-(100*self.time_slot_duration/1000))
        current_move = self.movement_matrix[np.nonzero(self.movement_matrix[:, 0].A == self.start_time)[0], :]
        for value in current_move:
            for i in range(1, 4):
                self.node_position[int(value[0, 1]), i] = value[0, i + 1]
        self.node_id_position = self.node_position[:, [1, 2, 3]]

        self.position_state = np.array(self.node_id_position)

        for node in self.node_list:
            node.update_node_position(self.node_id_position)
            node.generate_hello(self.controller)
        # 控制器更新网络全局情况
        self.controller.predict_position()
        ## 如果达到条件，对可调整控制器位置进行多目标ABC，获得开关情况
        self.controller.init_controller_place(2)
        self.control_state = []
        for i in self.controller:
            self.control_state.append(i.controller_position)
        self.control_state = np.array(self.control_state)

        b = np.zeros(self.vehicle_num)
        b1 = np.ones(self.control_num)
        self.position_state = np.insert(self.position_state, 3, values=b, axis=1)
        self.control_state = np.insert(self.control_state, 3, values=b1, axis=1)

        self.current_state = np.concatenate((self.position_state, self.control_state))
        self.terminal = False
        self.channel_penalty = 0
        self.power_penalty = 0
        self.flight_penalty = 0
        self.AoI_sum = 0
        self.throughput_sum = 0

    def step(self, action_1,  action_2, cnt):
        '''
            step one move and feed back reward
        '''
        # direction = {
        #     0: np.array([1, self.current_state[1]]),  # left
        #     1: np.array([1, self.current_state[1] + 1]),  # right
        # }[action]

        # self.current_state = self.current_state + direction
        # action_1 UAV movement
        
        time = self.start_time + self.time_slot_duration / 1000.0 * cnt

        current_move = self.movement_matrix[np.nonzero(self.movement_matrix[:, 0].A == int(time))[0], :]
        for value in current_move:
            for i in range(1, 4):
                self.node_position[int(value[0, 1]), i] = value[0, i + 1]
        self.node_id_position = self.node_position[:, [1, 2, 3]]
        
        next_move = self.movement_matrix[np.nonzero(self.movement_matrix[:, 0].A == int(time)+1)[0], :]
        for value in next_move:
            for i in range(1, 4):
                self.next_node_position[int(value[0, 1]), i] = value[0, i + 1]

        self.next_node_id_position = self.next_node_position[:, [1, 2, 3]]
            
        self.node_id_position = self.node_id_position+(self.next_node_id_position-self.node_id_position)*(time-int(time))

        self.position_state = np.array(self.node_id_position)

        for i in range(self.control_num):
            if action_1[i,0]**2 + action_1[i,1]**2 + action_1[i,2]**2 > (self.speed_limit*(self.time_slot_duration/1000))**2:
                self.flight_penalty += 1

        self.control_state = self.control_state + action_1

        # b = np.zeros(self.vehicle_num)
        # b1 = np.ones(self.control_num)

        # self.position_state = np.insert(self.position_state, 3, values=b, axis=1)

        # self.control_state = np.insert(self.control_state, 3, values=b1, axis=1)

        # self.current_state = np.concatenate((self.position_state, self.control_state))
         
        # self.c_v_co_association = action_2[:,:self.control_num]
        # ind = self.control_num
        # self.c_v_ch_association = action_2[:,ind:ind+self.channel_num]
        # ind+=self.channel_num
        # self.t_v_co_association = action_2[:,ind:ind+self.control_num]
        # ind+=self.control_num
        # self.t_v_ch_association = action_2[:,ind:ind+self.channel_num]
        # ind+=self.channel_num

        # self.c_v_co_association = np.eye(self.c_v_co_association.shape[1])[self.c_v_co_association.argmax(1)]
        # self.c_v_ch_association = np.eye(self.c_v_ch_association.shape[1])[self.c_v_ch_association.argmax(1)]
        # self.t_v_ch_association = np.eye(self.t_v_ch_association.shape[1])[self.t_v_ch_association.argmax(1)]
        # self.t_v_co_association = np.eye(self.t_v_co_association.shape[1])[self.t_v_co_association.argmax(1)]
        
        # self.c_v_power = (action_2[:,ind:ind+1]).T
        # ind+=1
        # self.t_v_power = (action_2[:,ind:ind+1]).T
        # ind+=1

        for i in range(self.vehicle_num):
            if np.argmax(self.c_v_co_association) == np.argmax(self.t_v_co_association) and np.argmax(self.c_v_ch_association) == np.argmax(self.t_v_ch_association):
                self.channel_penalty += 1
            if self.c_v_power[i] + self.t_v_power[i] > self.power_limit[i]:
                self.power_penalty += 1
    
        c_sinr, t_sinr = A2G_channel_1(self.c_v_co_association,self.c_v_ch_association,self.t_v_co_association,self.t_v_ch_association,self.c_v_power,self.t_v_power,self.position_state, self.control_state)
        for i in range(len(c_sinr)):
            if c_sinr[i] >= self.threshold:
                self.current_AoI[i] = int(1024*8/(math.log2(1 + c_sinr[i]) * band)/self.time_slot_duration)
            else:
                self.current_AoI[i] += 1
            self.AoI_sum += self.current_AoI[i]
        
        for i in range(len(t_sinr)):
            if t_sinr[i] >= self.threshold:
                self.throughput[i] = math.log2(1 + t_sinr[i]) * band
            else:
                self.throughput[i] = 0
            self.throughput_sum += self.throughput[i]
        ### Reward
        
        if cnt==100:
            self.terminal = True
            reward = (self.AoI_sum+self.throughput_sum)/100/self.vehicle_num+self.channel_penalty+self.power_penalty+self.flight_penalty


        return self.current_state, reward, self.terminal
