import pyautogui
import keyboard
import time

def main():
    print("O script vai começar em 5 segundos... Pressione ESC para cancelar.")
    time.sleep(5)

    keys = ['w', 'w', 'a', 'a', 'd', 'd', 's', 's', 'c', 'z', 'v']

    while not keyboard.is_pressed('esc'):
        for key in keys:
            if keyboard.is_pressed('esc'):
                print("Execução interrompida.")
                return
            pyautogui.press(key)
            time.sleep(0.1)
    print("Execução concluída.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Execução interrompida.")
