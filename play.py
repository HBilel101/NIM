from nim import train, play
import argparse
parser = argparse.ArgumentParser(prog="NIM AI Game",description="play with NIM AI agent")
parser.add_argument("-n",type=int,help='defines the number of epochs the model trains',default=10000)
args = parser.parse_args()
ai = train(args.n)
play(ai)
