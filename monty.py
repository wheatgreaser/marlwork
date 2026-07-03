class State:
    def __init__(self, value, next_state, goal_state, transition_rewards):
        self.next_state = next_state
        self.goal_state = goal_state
        self.value = value
        self.transition_rewards = transition_rewards

class Agent:
    current_state = None
    action = None
    total_reward = 0

goal1 = State(0, None, True, 0)
goal2 = State(0, None, True, 0)
state2 = State(0, [goal1, goal2], False, [10, 5])
state1 = State(0, state2, False, 2)

action_value_table = {"state2":0, "goal1":0, "goal2":0}





