import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["alphazero", "random"], default="alphazero")
    parser.add_argument("--size", type=int, default=5, help="Board size (for random agent)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes (for random agent)")
    parser.add_argument("--model-id", type=str, default="random", help="Log/model id (for random agent)")
    args = parser.parse_args()

    if args.agent == "random":
        from agents.random.random import RandomAgent
        agent = RandomAgent(model_id=args.model_id, size=args.size, episodes=args.episodes)
        agent.run()
    else:
        from agents.alphazero.alphazero import AlphaZero
        alphazero = AlphaZero()