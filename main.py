# PAYOFF_MATRIX = [[3, 0], [5, 1]]
# COOPERATE, BETRAY = 0, 1
# your_moves, opponent_moves = [], []
# your_score, opponent_score = 0, 0

# def reset():
#     your_moves, opponent_moves = [], []
#     your_score, opponent_score = 0, 0

# def next_turn(your_move, opponent_move):
#     your_moves.append(your_move)
#     opponent_moves.append(opponent_move)
#     your_score += PAYOFF_MATRIX[your_move][opponent_move]
#     opponent_score += PAYOFF_MATRIX[opponent_move][your_move]
#     return PAYOFF_MATRIX[your_move][opponent_move]

# def play(round, your_strategy, opponent_strategy):
#     reset()
#     for i in range(round):
#         next_turn(your_strategy(), opponent_strategy())

import neat, os, random

tictactoe_inputs = [tuple([random.choice([-1,0,1]) for i in range(9)]) for j in range(25)]
tictactoe_outputs = [[int(i==0) for i in board] for board in tictactoe_inputs]

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        numberOfBoards = float(len(tictactoe_inputs))
        genome.fitness = 1.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)


        for xi, xo in zip(tictactoe_inputs, tictactoe_outputs):
            output = net.activate(xi)

            for i in range(len(output)):
                genome.fitness -= (float(xo[i] - output[i]) / float(len(output))) ** 2

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(5))

# Run for up to 300 generations.
winner = p.run(eval_genomes, 300)

# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))

# Show output of the most fit genome against training data.
print('\nOutput:')
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
for xi, xo in zip(tictactoe_inputs, tictactoe_outputs):
    output = winner_net.activate(xi)
    print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
