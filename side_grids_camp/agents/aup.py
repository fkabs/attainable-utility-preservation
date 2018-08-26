import copy
import numpy as np
from ai_safety_gridworlds.environments.shared.safety_game import Actions


class AUPAgent():
    """
    Attainable utility-preserving agent.
    """
    name = 'Attainable Utility Preservation'
    null_action = 4  # TODO make robust

    def __init__(self, penalties=(), m=6):
        """

        :param penalties: the estimators whose shifts in attainable value will be penalized.
        :param m: the horizon up to which the agent will act after acting.
        """
        self.penalties = penalties
        self.m = m

    def act(self, env, actions=[]):
        """Get penalties from brute-force search and choose best penalized action.

        :param actions: the actions up until now (assuming a deterministic environment, this allows re
        """
        penalized_rewards = self.penalized_rewards(env, actions)
        return np.argmax(penalized_rewards)

    def restart(self, env, actions):
        """Reset the environment and return the result of executing the action sequence."""
        env.reset()
        for action in actions:
            env.step(action)
        return env

    def penalized_rewards(self, env, actions=[]):
        rewards, penalties = [], []  # the best attainable rewards and penalties after each action

        for action in range(env.action_spec().maximum + 1):
            # Set environment appropriately and get action's result
            if action > 0:
                self.restart(env, actions)
            time_step = env.step(action)

            # Get the attainable rewards within m steps after taking this action
            next_reward, next_pens = self.attainable_rewards(time_step, env, self.m, actions=actions + [action],
                                                             visited=set([str(time_step.observation['board'])]))
            rewards.append(next_reward)
            penalties.append(np.array(next_pens))

        # Difference of attainable rewards between taking action and doing nothing
        action_differences = np.array([abs(penalty - penalties[self.null_action])
                                       for penalty in penalties])
        weighted_penalties = np.array([sum(diffs) / len(diffs) for diffs in action_differences]) if self.penalties \
            else np.zeros(len(rewards))
        return np.array(rewards) - weighted_penalties

    def attainable_rewards(self, time_step, env, steps_left, actions=[], visited=set()):
        """Returns best normal and penalty rewards attainable within steps_left steps.

        :param time_step: TimeStep object containing last observation and reward.
        :param env: Simulator.
        :param steps_left: Remaining depth.
        :param actions: Actions taken up until now.
        :param visited: States already visited.
        """
        pens = [penalty(time_step.observation) for penalty in self.penalties]  # TODO check multiple times attaining goal reward off-goal?
        if steps_left == 0 or time_step.last():
            return time_step.reward if time_step.reward is not None else 0, pens

        rewards, penalty_lsts = [], []  # for each action, how much reward can we get?
        for action in range(env.action_spec().maximum + 1):
            if action > 0:  # don't need to reset the first time
                env = self.restart(env, actions)
            #new_env = copy.deepcopy(env)
            # Take a new action and see if we've been there; if so, add to visited.
            new_time_step = env.step(action)
            if str(new_time_step.observation['board']) in visited:  # don't revisit these states
                continue
            visited.add(str(new_time_step.observation['board']))

            # See what reward and penalties we can attain from here
            reward, next_pens = self.attainable_rewards(new_time_step, env, steps_left-1,
                                                        actions=actions + [action], visited=visited)

            visited.remove(str(new_time_step.observation['board']))
            rewards.append(reward)
            penalty_lsts.append(next_pens)

        if len(rewards) == 0 and len(penalty_lsts) == 0:
            return 0, np.zeros(len(self.penalties))  # in case every new state was already in visited

        return (time_step.reward if time_step.reward is not None else 0) + max(rewards), \
               [pen + max(next_pens) for pen, next_pens in zip(pens, penalty_lsts)]
