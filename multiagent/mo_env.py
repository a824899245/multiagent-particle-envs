from __future__ import absolute_import, division, print_function
import numpy as np

from .SDVN import SDVN


class MultiObjectiveEnv(object):

    def __init__(self, env_name="deep_sea_treasure"):
        if env_name == "SDVN":
            self.env = SDVN()
            self.state_spec = self.env.state_n
            self.action_spec = self.env.action_n
            self.reward_spec = self.env.reward_n


    def reset(self, env_name=None):
        ''' reset the enviroment '''
        self.env.reset()

    def observe(self):
        ''' reset the enviroment '''
        return self.env.current_state

    def step(self, action, cnt):
        ''' process one step transition (s, a) -> s'
            return (s', r, terminal)
        '''
        return self.env.step(action, cnt)


if __name__ == "__main__":
    '''
        Test ENVs
    '''
    dst_env = MultiObjectiveEnv("ft7")
    dst_env.reset()
    terminal = False
    print("DST STATE SPEC:", dst_env.state_spec)
    print("DST ACTION SPEC:", dst_env.action_spec)
    print("DST REWARD SPEC:", dst_env.reward_spec)
    while not terminal:
        state = dst_env.observe()
        action = np.random.choice(2, 1)[0]
        next_state, reward, terminal = dst_env.step(action)
        print("s:", state, "\ta:", action, "\ts':", next_state, "\tr:", reward)
    print("AN EPISODE ENDS")
