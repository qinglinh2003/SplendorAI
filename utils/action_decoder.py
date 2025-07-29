from utils.valid_action_utils import enumerate_all_token_actions
from env.splendor_env import SplendorEnv
import numpy as np

def action_to_instruction(env:SplendorEnv, action:int):
    action = int(action)
    action_type, action_param = env.decode_action(action)
    if action_type == "take_token":
        final = enumerate_all_token_actions()[action_param][2][:5]
        action_decoded = zip(["Green", "White", "Blue", "Black", "Red"], final)
        return ("take_token",
                str([number for gem, number in enumerate(action_decoded)]))

    elif action_type == "buy_card":
        row, index = divmod(action_param, 4)
        return ("buy_card",
                f'Tier:{row}, Index:{index}'
                )

    elif action_type == "reserve_card":
        row, index = divmod(action_param, 4)
        return ("reserve_card",
                f'Tier:{row}, Index:{index}'
                )

    elif action_type == "buy_reserved_card":
        return ("buy_reserve",
                f"Index:{action_param}"
        )

    else:
        raise Exception(f"Unknown action type: {action_type}")



def action_to_click_sequence(env:SplendorEnv, action:int):
    action = int(action)
    action_type, action_param = env.decode_action(action)
    all_token_actions = enumerate_all_token_actions()
    click_sequence = []
    if action_type == "take_token":
        click_sequence.append("take_token")
        click_sequence.append(False) # False for not returning gems
        take, discard, final  = all_token_actions[action_param]
        color_map = {
            0: "C",
            1: "S",
            2: "E",
            3: "R",
            4: "O",
        }

        for index, num in enumerate(take):
            for i in range(num):
                click_sequence.append(color_map[index])

        discard_sequence = []
        if np.sum(discard) > 0:
            click_sequence[1] = True
            for index, num in enumerate(discard):
                for i in range(num):
                    discard_sequence.append(color_map[index])
        click_sequence.append(discard_sequence)

    elif action_type == "buy_card":
        click_sequence.append("buy_card")
        click_sequence.append(action_param)


    return click_sequence



if "__main__" == __name__:
    env = SplendorEnv()
    print(action_to_click_sequence(env, 1483))