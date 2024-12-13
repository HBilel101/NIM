import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress info and warnings

from nim import PolicyGradientAI, Nim  # Import your AI and game logic
import random 
def play(ai, human_player=None):
    if human_player is None:
        human_player = random.randint(0, 1)
    game = Nim()

    while True:
        print("\nPiles:")
        for i, pile in enumerate(game.piles):
            print(f"Pile {i}: {pile}")

        if game.player == human_player:
            print("Your Turn")
            while True:
                try:
                    pile = int(input("Choose Pile: "))
                    count = int(input("Choose Count: "))
                    if (pile, count) in Nim.available_actions(game.piles):
                        action = (pile, count)
                        break
                    else:
                        print("Invalid move, try again.")
                except ValueError:
                    print("Invalid input, please enter numbers only.")
        else:
            print("AI's Turn")
            state = game.piles
            action = ai.choose_action(state, epsilon=0)
            print(f"AI chose to take {action[1]} from pile {action[0]}")

        # Make move
        game.move(action)

        # Check for winner
        if game.winner is not None:
            print("\nGAME OVER")
            print(f"Winner is {'Human' if game.winner == human_player else 'AI'}")
            return

if __name__ == "__main__":
    ai = PolicyGradientAI()  # Initialize your AI
    ai.train(1000)           # Train the AI with 1000 games
    play(ai)                 # Play against the trained AI
