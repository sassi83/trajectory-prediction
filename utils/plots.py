import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches

def plot_fov(traj, rois, enviro):
    fig, ax = plt.subplots()
    
    enviro = enviro * 255
    [index] = np.random.randint(len(rois), size=1)
    ax.imshow(enviro.astype(int))
    for i, roi in enumerate(rois[index]):
        xl, yl, xr, yr = roi
        xl, yl = int(xl*enviro.shape[1]), int(yl*enviro.shape[0])
        w, h = int(xr*enviro.shape[1]-xl), int(yr*enviro.shape[0]-yl)
        rect = patches.Rectangle((xl,yl),w,h,linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    ax.plot(traj[index, :, 0], traj[index, :, 1], color='w', marker='s', markersize=2)
    plt.show()
    

def plot_environ(enviro, name=None):
    extent = [0, enviro.shape[1], enviro.shape[0], 0]
    fig = plt.figure()
    ax0 = fig.add_subplot(131)
    ax1 = fig.add_subplot(132)
    ax2 = fig.add_subplot(133)
    ax0.set_aspect("equal")
    ax1.set_aspect("equal")   
    ax2.set_aspect("equal") 
    ax0.imshow(enviro[:, :, 0], extent=extent, origin='upper', cmap=cm.jet, vmin=0, vmax=1)
    ax1.imshow(enviro[:, :, 1], extent=extent, origin='upper', cmap=cm.jet, vmin=0, vmax=1)
    ax2.imshow(enviro[:, :, 2], extent=extent, origin='upper', cmap=cm.jet, vmin=0, vmax=1) 
    ax0.set_title("Trajectory")
    ax1.set_title("Orientaion")
    ax2.set_title("Speed")
    plt.show()
    plt.gcf().clear()
    plt.close()    
  

def plot_maps(map, img_extent, des, dataname=None):
    fig, ax = plt.subplots()
    ax.set_aspect("equal")    
    img = ax.imshow(map, extent=img_extent, origin='upper', cmap=cm.jet)
    plt.colorbar(img, ax=ax)
    ax.set_title(des)
    # plt.savefig("fig/test_real/%s_speed_map.png"%(dataname), bbox_inches='tight', dpi=300)
    plt.show()
    plt.gcf().clear()
    plt.close("all")
    
    
def plot_pred(xy, y_prime, N=10, save=False, index=0, base_path="images/"):

    fig, ax = plt.subplots()
    pred_seq = y_prime.shape[2]
    obs_seq = xy.shape[1] - pred_seq
    

    for i in range(N):
        # plot observation
        ax.plot(xy[i, :obs_seq, 0], xy[i, :obs_seq, 1], color='k', alpha=1.0)
        # plot ground truth
        ax.plot(xy[i, obs_seq-1:, 0], xy[i, obs_seq-1:, 1], color='r', alpha=1.0)
        for j, pred in enumerate(y_prime[i]):
            # concate the first step for visulization purpose
            #pred = np.concatenate((xy[i, obs_seq-1:obs_seq], pred), axis=0)
            ax.plot(pred[:, 0], pred[:, 1], color='b', alpha=0.2)

    ax.set_aspect("equal")
    if save:
        save_folder = base_path + "trajectory_index_" + str(index) + ".jpg"
        plt.savefig(save_folder, bbox_inches='tight', dpi=150)
        plt.gcf().clear()
        plt.close()
    else:
        plt.show()
        plt.gcf().clear()
        plt.close()


def plot_error_bar(xy_truth=None, predictions=None, std=None, batches=10, agents=2, args=None, save=False, base_path="images/"):

    for batch in range(batches):
        for agent in range(agents):
            save_folder = base_path + "trajectory_batch_" + str(batch) + "_agent_" + str(agent) + ".jpg"
            plt.figure()
            # plot history
            plt.errorbar(xy_truth[batch, :args.obs_seq, 2], xy_truth[batch, :args.obs_seq, 3], c='black')
            # plot ground truth
            plt.errorbar(xy_truth[batch, args.obs_seq - 1:, 2], xy_truth[batch, args.obs_seq - 1:, 3], c='red')
            # plot prediction
            plt.errorbar(predictions[batch, agent, :, 0], predictions[batch, agent, :, 1], xerr=std[batch, agent, :, 0],
                         yerr=std[batch, agent, :, 1], c='blue', fmt='o', ecolor='g', capthick=2, alpha=0.2)


            if save:
                plt.savefig(save_folder, bbox_inches='tight', dpi=150)
                plt.gcf().clear()
                plt.close()


            else:
                plt.show()
                plt.gcf().clear()
                plt.close()



def plot_dist(xy_truth=None, predictions=None, std=None, args=None, save=False, base_path="images/"):
    t = np.linspace(0, 2 * np.pi, 100)
    for batch in range(args.batch_size):
        for agent in range(args.agents):
            save_folder = base_path + "trajectory_batch_" + str(batch) + "_agent_" + str(agent) + ".jpg"
            plt.figure()
            # plot history
            plt.plot(xy_truth[batch, :args.history, 0], xy_truth[batch, :args.history, 1], c='black')
            # plot ground truth
            plt.plot(xy_truth[batch, args.history - 1:, 0], xy_truth[batch, args.history - 1:, 1], c='red')

            # plot prediction
            plt.plot(predictions[batch, agent, :, 0], predictions[batch, agent, :, 1], c='blue')
            for point in range(args.future):
                Ell = np.array([std[batch, agent, point, 0] * np.cos(t), std[batch, agent, point, 1] * np.sin(t)])
                plt.plot(predictions[batch, agent, point, 0] + Ell[0, :], predictions[batch, agent, point, 1] + Ell[1, :], c='blue', alpha=0.2)


            if save:
                plt.savefig(save_folder, bbox_inches='tight', dpi=150)
                plt.gcf().clear()
                plt.close()


            else:
                plt.show()
                plt.gcf().clear()
                plt.close()

