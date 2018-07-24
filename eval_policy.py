import re


class EvalPolicy:

    def __init__(self):
        self.env_actions_rewards = dict()

    def list_to_str(self, values):
        actions = ['None', 'forward', 'right', 'left']
        clean_values = map(lambda value: value.replace('\n', ''), values)
        clean_values = map(lambda value: float(value.split(':')[1]), clean_values)
        return actions[clean_values.index(max(clean_values))]

    def get_action_reward(self, lines):
        status = lines[0].split(':')[1].strip()
        values = lines[1].split('rewarded')
        description = values[0].replace('(', '').strip()
        reward = values[1].replace(')', '').strip()
        return {'status': status, 'description': description, 'reward': float(reward)}

    def process_action_reward(self, record):
        if record['status'] in self.env_actions_rewards:
            current_reward = self.env_actions_rewards[record['status']]['reward']
            if record['reward'] > current_reward:
                self.env_actions_rewards[record['status']] = {'description': record['description'], 'reward': record['reward']}
        else:
            self.env_actions_rewards[record['status']] = {'description': record['description'], 'reward': record['reward']}

    def get_sim_rewards(self, lines):
        self.env_actions_rewards = dict()
        total_lines = len(lines)
        for i in range(total_lines):
            if "Agent previous state:" in lines[i]:
                record = self.get_action_reward(lines[i: i + 2])
                self.process_action_reward(record)
        return self.env_actions_rewards

    def get_policy(self):
        text_file = open("logs/sim_improved-learning.txt", "r")
        lines = text_file.readlines()
        line_map = enumerate(lines)
        states_map = map(lambda state: (state[0], state[1].replace('\n', '')), filter(lambda state: re.match(r'([a-zA-Z]+)-([a-zA-Z]+)-([a-zA-Z]+)-([a-zA-Z]+)-([a-zA-Z]+)', state[1]), line_map))
        states_map = map(lambda state: (state[1], self.list_to_str(lines[state[0] + 1: state[0] + 5])), states_map)
        return dict(states_map)

    def is_waypoint_blocked(self, waypoint, oncoming, left):
        if waypoint == 'right':
            if left == 'forward':
                return True
            else:
                return False
        elif waypoint == 'left' and (oncoming == 'right' or oncoming == 'forward'):
            return True
        elif waypoint == 'forward':
            return False
        else:
            return False

    def eval_state(self, state):
        waypoint, light, oncoming, right, left = state.split('-')

        # I have green light but my waypoint is blocked, I choose the next available route
        if light == 'green':
            if waypoint == 'right' or waypoint == 'forward':
                return waypoint
            elif waypoint == 'left' and not self.is_waypoint_blocked(waypoint, oncoming, left):
                return 'left'
            elif waypoint == 'left' and self.is_waypoint_blocked(waypoint, oncoming, left):
                if oncoming == 'right':
                    return None
                elif oncoming == 'forward':
                    return 'right'
            else:
                return None
        elif light == 'red' and waypoint == 'right' and not self.is_waypoint_blocked(waypoint, oncoming, left):
            return 'right'
        elif light == 'red' and self.is_waypoint_blocked(waypoint, oncoming, left):
            return None
        return None

    def compare_policy(self):
        import pandas as pd

        policy_map = self.get_policy()
        policy_df = pd.DataFrame(policy_map.items(), columns=['State', 'Action'])

        states = map(lambda policy: policy.split(',')[0], policy_map)
        optimal_policy = dict(map(lambda state: (state, str(self.eval_state(state))), states))
        optimal_policy_df = pd.DataFrame(optimal_policy.items(), columns=['State', 'Action'])

        policy_comparison_df = policy_df.merge(optimal_policy_df,  on='State', how='inner')
        return policy_comparison_df.rename(index=str, columns={"Action_x": "Agent Action", "Action_y": "Optimal Action"})


if __name__ == '__main__':
    import pandas as pd

    state = 'right-red-right-None-left'
    evalPolicy = EvalPolicy()
    print evalPolicy.eval_state(state)
