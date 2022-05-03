import pandas as pd

from p5.track import Track
from p5.settings import ACTIONS, GAMMA, P_ACC_SUCCEED, P_ACC_FAIL
from p5.utils import bresenham, compute_velocity


def compute_state_weights(action_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute weights / transition probabilities for one state.
    :param action_df: Table of actions
    :return: Updated actions table that includes state weights / transition probabilities
    """
    mask = action_df["s"]
    succeed_tot = mask.sum()
    fail_tot = len(action_df) - succeed_tot
    action_df.loc[mask, "wt"] = P_ACC_SUCCEED * action_df.loc[mask, "ct"] / succeed_tot
    action_df.loc[~mask, "wt"] = P_ACC_FAIL * action_df.loc[mask, "ct"] / fail_tot
    return action_df


def learn_state(track: Track, states, ix, oob_penalty) -> dict:
    """
    Learn one state.
    :param track: Track object
    :param states: Dataframe of states
    :param ix: States dataframe index
    :param oob_penalty: Out-of-bounds penalty
    :return: Best value and action
    """
    # Extract a dictionary of state attributes: space, position, velocity, reward, time, etc.
    state_di = states.loc[ix].to_dict()
    vel = state_di["x_col_vel"], state_di["y_row_vel"]
    best_action, best_Q_sa = None, -float("inf")
    for action in ACTIONS:
        # Case when action succeeds
        succeed_vel = compute_velocity(vel, action)
        succeed_pos_li: list = update_position(state_di, track, succeed_vel, oob_penalty=oob_penalty, succeed=True)

        # Case when action fails
        fail_vel = compute_velocity(vel, (0, 0))
        fail_pos_li: list = update_position(state_di, track, fail_vel, oob_penalty=oob_penalty, succeed=False)

        # Combine success and failure cases
        actions = pd.concat([pd.DataFrame(succeed_pos_li), pd.DataFrame(fail_pos_li)])
        actions = compute_state_weights(actions)
        di = {"x_col_vel_1": "x_col_vel", "y_row_vel_1": "y_row_vel", "x_col_pos_1": "x_col_pos",
              "y_row_pos_1": "y_row_pos"}
        labels = ["x_col_vel", "y_row_vel", "x_col_pos", "y_row_pos", "val"]
        on = [x for x in labels if x != "val"]
        actions: pd.DataFrame = actions.rename(columns=di).merge(states[labels], on=on, how="left")

        # Compute the expected value
        Q_sa: float = state_di["r"] + GAMMA * (actions.wt * actions.val).sum()
        if Q_sa > best_Q_sa:
            best_Q_sa = Q_sa
            best_action = action
    best_dict = dict(val=best_Q_sa, best_x_col_a=best_action[0], best_y_row_a=best_action[1])
    return best_dict


def update_position(state_di: dict, track: Track, vel_1: tuple, oob_penalty="stay-in-place",
                    succeed=True) -> list:
    """
    Update position based on velocity.
    """
    x_col_vel_1, y_row_vel_1 = vel_1
    x_col_pos_0, y_row_pos_0 = state_di["x_col_pos"], state_di["y_row_pos"]
    x_col_pos_1 = x_col_pos_0 + x_col_vel_1
    y_row_pos_1 = y_row_pos_0 + y_row_vel_1
    reward = track.get_reward((x_col_pos_1, y_row_pos_1))

    positions = []
    for pos in bresenham(x_col_pos_0, y_row_pos_0, x_col_pos_1, y_row_pos_1):

        # Case when driver hits wall
        if pos in track.wall_cells:

            # Case when OOB (out-of-bounds) penalty is to remain in current position
            if oob_penalty == "stay-in-place":
                x_col_pos_1 = x_col_pos_0
                y_row_pos_1 = y_row_pos_0
                reward = track.get_reward((x_col_pos_1, y_row_pos_1))
                di = dict(x_col_vel_1=x_col_vel_1, y_row_vel_1=y_row_vel_1, x_col_pos_1=x_col_pos_1,
                          y_row_pos_1=y_row_pos_1, ct=1, fin=False, r=reward, s=succeed)
                positions.append(di)

            # Case when OOB penalty is to return to one of the start positions
            else:
                prob = 1 / len(track.start_cells)
                for start_pos in track.start_cells:
                    reward = track.get_reward(start_pos)
                    di = dict(x_col_vel_1=x_col_vel_1, y_row_vel_1=y_row_vel_1, x_col_pos_1=start_pos[0],
                              y_row_pos_1=start_pos[1], ct=prob, fin=False, r=reward,
                              s=succeed)
                    positions.append(di)

            return positions

        # Case when driver reaches finish line
        elif pos in track.finish_cells:
            x_col_pos_1, y_row_pos_1 = pos
            reward = track.get_reward(pos)
            di = dict(x_col_vel_1=x_col_vel_1, y_row_vel_1=y_row_vel_1, x_col_pos_1=x_col_pos_1,
                      y_row_pos_1=y_row_pos_1, ct=1, fin=True, r=reward, s=succeed)
            positions.append(di)
            # Stop if driver reaches finish line
            return positions

    # Case when driver reaches a non-wall destination
    fin = True if (x_col_pos_1, y_row_pos_1) in track.finish_cells else False
    positions.append(
        dict(x_col_vel_1=x_col_vel_1, y_row_vel_1=y_row_vel_1, x_col_pos_1=x_col_pos_1, y_row_pos_1=y_row_pos_1, ct=1,
             fin=fin, r=reward, s=succeed))
    return positions
