import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from agents.ppo_agent import PPOAgent
from agents.rule_based_agent import RuleBasedAgent
from env.splendor_env import SplendorEnv
from sb3_contrib import MaskablePPO

from state_detection.test_selenium import execute_click_sequence
from utils.action_decoder import action_to_instruction, action_to_click_sequence
from state_detection.parse_web import get_obs, get_cur_turn


USE_RULE_BASED = True
MODEL_PATH = "../agents/checkpoints/checkpoint_step_400000.zip"
TABLE_ID = "639738969"
USERNAME = "97495998"
SLEEP_TIME = 5

if USE_RULE_BASED:
    ai_agent = RuleBasedAgent()
else:
    model = MaskablePPO.load(MODEL_PATH)
    ai_agent = PPOAgent(model)




def main_loop():
    env = SplendorEnv()  #
    reserved_cards = []
    last_turn = 0
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)

    driver.get("https://boardgamearena.com")
    input(" Game Entered...")


    while True:
        cur_turn = get_cur_turn()


        if cur_turn == 0 or cur_turn != last_turn:
            print("\n Getting the latest game state...")
            obs = get_obs(USERNAME)
            print(obs)
            print(" AI predicts the best move...")
            action = ai_agent.act(obs)


            action_type, action_param = env.decode_action(action)

            try:
                click_sequence = action_to_click_sequence(env, action)
                print(click_sequence)
                execute_click_sequence(driver, click_sequence)
            except Exception as e:
                print(e)

            if action_type == "reserve_card":
                reserved_cards.append(obs['board_cards'][action_param])
            elif action_type == "buy_reserved_card":
                if 0 <= action_param < len(reserved_cards):
                    reserved_cards.pop(action_param)
                else:
                    print(f" Index {action_param} our of bound for reserved_cards ")

            print(" Decoding AI's Choice...")
            action_text = action_to_instruction(env, action)
            print(f"\n **AI Recommends：** {action_text}")


            time.sleep(SLEEP_TIME)
            last_turn = cur_turn

        else:

            time.sleep(SLEEP_TIME)


if __name__ == "__main__":
    main_loop()
