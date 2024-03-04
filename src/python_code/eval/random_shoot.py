from src.python_code.OAMP.eval.main_eval import inference
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser()
# DoughRolling, DoughCutting, DoughGathering, DoughShaping
parser.add_argument("--task", type=str, default="ClothHang")
# Available agent: SoftGPT, GraphPPO, GraphSAC, PPO, SAC
parser.add_argument("--agent", type=str, default="ORK")
parser.add_argument("--policy", type=str, default="ppo")  # Available policy agent: ppo, sac, unless agent=SoftGPT
parser.add_argument("--mode", type=str, default="Particle")  # Available modes: Particle, Image, Graph
parser.add_argument("--sim_device", type=int, default=0)
parser.add_argument("--rl_device", type=int, default=0)
parser.add_argument("--inference", action="store_false", help="turn to inference mode while adding this argument")
parser.add_argument("--ckpt_path", type=str, default="/home/ubuntu/Github/ManiCloth/src/python_code/OAMP/runs/RofuncRL_ORKTrainer_ClothHang_24-02-27_09-53-01-768486/checkpoints/best_ckpt.pth")
custom_args = parser.parse_args()

# new = inference(custom_args, np.array([10, 12, 13]))
# print(new)

exploration_steps = 500

init_bar_pose = np.array([-5.29989, 6.811906, -0.285244])
bar_pose = []
for i in range(exploration_steps):
    bar_pose.append(init_bar_pose)
bar_pose = np.array(bar_pose)

random = np.random.rand(exploration_steps, 3) * 20
random = random - 10
random = random + bar_pose

# print(random)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Z-axis')
# ax.set_zlabel('Y-axis')

bar_pos = []
fall = []

for i in range(exploration_steps):
    x, v = inference(custom_args, random[i], render=False)
    x = x[-1]

    y_value = []
    for j in range(len(x)):
        y_value.append(x[j, 1])

    y_value = np.array(y_value)

    if np.mean(y_value) < 5.5:
        fall.append(True)
    else:
        fall.append(False)

    bar_pos.append(random[i])

    if fall[i] == True:
        print("Fallen at step: ", i)

    if fall[i] == False:
        print("Success at step: ", i)

    print("Bar Position is", random[i])
    print("Currently at step: ", i)

    # ax.scatter(-5.29989, -0.285244, 6.811906, c='b', marker='o')
    #
    # if fall[i] == True:
    #     ax.scatter(bar_pos[i][0], bar_pos[i][2], bar_pos[i][1], c='r', marker='o')
    # else:
    #     ax.scatter(bar_pos[i][0], bar_pos[i][2], bar_pos[i][1], c='g', marker='o')
    #
    # plt.show(block=False)
    # plt.pause(3)
    # plt.close()

"""
Save the graph
"""

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(exploration_steps):
    # Init point of the bar
    ax.scatter(-5.29989, -0.285244, 6.811906, c='b', marker='o')

    if fall[i] == True:
        ax.scatter(bar_pos[i][0], bar_pos[i][2], bar_pos[i][1], c='r', marker='o')
    else:
        ax.scatter(bar_pos[i][0], bar_pos[i][2], bar_pos[i][1], c='g', marker='o')


ax.set_xlabel('X-axis')
ax.set_ylabel('Z-axis')
ax.set_zlabel('Y-axis')

plt.savefig('/home/ubuntu/Github/ManiCloth/src/python_code/eval/eval_graphs/inference_plot_500.png')
plt.show()

# import pickle
# pickle.dump(fig, open('FigureObject.fig.pickle', 'wb'))