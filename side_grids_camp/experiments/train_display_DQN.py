from __future__ import print_function
from collections import namedtuple
from environment_helper import *
from ai_safety_gridworlds.environments.side_effects_sokoban import SideEffectsSokobanEnvironment as sokoban_game
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def plot_images_to_ani(framesets):
    """
    Animates all agent executions and returns the animation object.

    :param framesets: [("agent_name", frames),...]
    """
    fig, axs = plt.subplots(1, len(framesets), figsize=(5, 5 * len(framesets)))

    max_runtime = max([len(frames) for _, frames in framesets])

    ims, zipped = [], zip(framesets, axs if len(framesets) > 1 else [axs])  # handle 1-agent case
    for i in range(max_runtime):
        ims.append([])
        for (agent_name, frames), ax in zipped:
            if i == 0:
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.set_xlabel(agent_name)
            ims[-1].append(ax.imshow(frames[min(i, len(frames) - 1)], animated=True))
    return animation.ArtistAnimation(plt.gcf(), ims, interval=250, blit=True, repeat_delay=0)


start_time = datetime.datetime.now()
global_step = tf.train.create_global_step()
game, kwargs = sokoban_game, {'level': 0}

# Plot setup
plt.switch_backend('TkAgg')
plt.style.use('ggplot')
scores = plt.figure()
plt.ylabel("Score")
plt.xlabel("Episode")

with tf.Session() as sess:
    agents, envs = generate_agents_environments(sess, game, kwargs)

    num_episodes = 200
    EpisodeStats = namedtuple("EpisodeStats", ["lengths", "rewards", "performance"])
    stats_dims = (len(agents), num_episodes)
    stats = EpisodeStats(lengths=np.zeros(stats_dims), rewards=np.zeros(stats_dims),
                         performance=np.zeros(stats_dims))

    movies = []
    for i_agent, (agent, env) in enumerate(zip(agents, envs)):
        print("Beginning training of agent #{}.".format(i_agent))

        for i_episode in range(num_episodes):
            stats.rewards[i_agent, i_episode], stats.lengths[i_agent, i_episode], \
                stats.performance[i_agent, i_episode], _ = run_episode(agent, env)
            print("\rEpisode {}/{}, reward: {}".format(i_episode + 1, num_episodes,
                                                       stats.rewards[i_agent, i_episode]), end="")
        print("\n")

        agent.save()
        plt.plot(range(num_episodes), stats.rewards[i_agent])  # plot performance

        _, _, _, frames = run_episode(agent, env, epsilon=0, save_frames=True, render=True)  # get frames from final policy
        movies.append((agent.name, frames))
    plt.figure(scores.number)
    plt.show()  # show performance

    print("\nTraining finished for {}; {} elapsed.".format(game.name, datetime.datetime.now() - start_time))
    ani = plot_images_to_ani(movies)
    ani.save(os.path.join('side_grids_camp', 'gifs', sokoban_game.name + '.gif'), writer='imagemagick')
    plt.show()



