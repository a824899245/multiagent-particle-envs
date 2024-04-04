import numpy as np
from multiagent.SDVN_core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from Get_Move import get_position
from Init import init_controller, init_node
import math

class Scenario(BaseScenario):
    def make_world(self, vehicle_num=100, control_num = 4, channel_num = 4):
        world = World()
        world.reward_dim = 2
        world.vehicle_num = vehicle_num
        world.control_num = control_num
        world.channel_num = channel_num
        world.dt = 200
        # simulation start time
        world.start_time = 100
        # simulation slot
        world.slot = 0

        world.landmarks = [Landmark() for i in range(world.vehicle_num)]
        for i, landmark in enumerate(world.landmarks):
            landmark.id = i
            landmark.name = 'Vehicle %d' % i
        
        world.agents = [Agent() for i in range(world.control_num)]
        for i, agent in enumerate(world.agents):
            agent.id = i
            agent.name = 'Controller %d' % i

            
        input = "envs/rural_small_"+str(world.vehicle_num)+".mobility.tcl"
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

        world.movement_matrix, world.init_position_matrix = get_position('tiexi1.tcl')

        self.reset_world(world, world.slot)
        return world

        self.reset_world(world)
        return world

    def reset_world(self, world, slot):

        world.slot = slot
        
        time = world.start_time + world.dt / 1000.0 * world.slot

        current_move = np.copy(world.movement_matrix[np.nonzero(world.movement_matrix[:, 0].A == int(time))[0], :])

        node_position = np.copy(self.init_position_arranged[0])
        next_node_position = np.copy(self.init_position_arranged[0])

        for value in current_move:
            for i in range(1, 4):
                node_position[int(value[0, 1]), i] = np.copy(value[0, i + 1])

        node_id_position = np.copy(world.node_position[:, [1, 2, 3]])
        
        next_move = np.copy(world.movement_matrix[np.nonzero(world.movement_matrix[:, 0].A == int(time)+1)[0], :])
        for value in next_move:
            for i in range(1, 4):
                next_node_position[int(value[0, 1]), i] = np.copy(value[0, i + 1])
        
        next_node_id_position = np.copy(next_node_position[:, [1, 2, 3]])
            
        node_id_position = node_id_position+(next_node_id_position-node_id_position)*(time-int(time))


        for v in world.landmarks:
            v.pos = [node_id_position[v.id][0, 0], node_id_position[v.id][0, 1],
                         node_id_position[v.id][0, 2]]
            
        on_controller = []
        delta_x = 3350.24 / (math.sqrt(world.control_num)*2)
        delta_y = 2860.33  / (math.sqrt(world.control_num)*2)
        x = delta_x
        y = delta_y
        for i in range(math.sqrt(world.control_num)):
            for j in range(math.sqrt(world.control_num)):
                a = [x + delta_x * 2 * i, y + delta_y * 2 * j, 200]
                on_controller.append(a)

        for i, agent in enumerate(world.agents):
            agent.pos = on_controller[i]
        
        world.terminal = False

        # state space specification: 2-dimensional discrete box
        world.state_n = (world.control_num + world.vehicle_num) * 3 + world.vehicle_num

        # action space specification: 0 left, 1 right
        world.action_n = world.vehicle_num * world.channel_num*2 + world.vehicle_num * world.control_num*2 + world.control_num *3+ world.vehicle_num*2

        # reward specification: 2-dimensional reward
        # 1st: treasure value || 2nd: time penalty
        world.reward_n = 2

        # make initial conditions

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)