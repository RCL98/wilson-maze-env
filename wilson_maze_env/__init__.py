from gymnasium.envs.registration import register

register(
     id="WilsonMaze-v0",
     entry_point="wilson_maze_env.envs:WilsonMazeEnv",
)
