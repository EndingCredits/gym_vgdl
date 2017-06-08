from gym.envs.registration import registry, register, make, spec
from gym_vgdl.vgdl_env import VGDLEnv
import os

DATA_DIR = os.path.dirname(__file__) + '/vgdl/games/' 

register(
    id='VGDL-v0',
    entry_point='gym_vgdl:VGDLEnv',
    timestep_limit=1000,
    nondeterministic=True,
)

for game in ['aliens', 'boulderdash', 'chase', 'frogs', 'missilecommand', 'portals', 'sokoban', 'survivezombies', 'zelda']:
    register(
        id='vgdl_{}-v0'.format(game),
        entry_point='gym_vgdl:VGDLEnv',
        kwargs={
            'game_file': DATA_DIR + game + '.txt',
            'map_file': DATA_DIR + game + '_lvl0.txt',
            'obs_type': 'image'
        },
        timestep_limit=1000,
        nondeterministic=True,
    )


register(
    id='vgdlobstest-v0',
    entry_point='gym_vgdl:VGDLEnv',
    kwargs={
        'game_file': DATA_DIR + 'aliens' + '.txt',
        'map_file': DATA_DIR + 'aliens' + '_lvl0.txt',
        'obs_type': 'objects'
    },
    timestep_limit=1000,
    nondeterministic=True,
)

register(
    id='vgdltest-v0',
    entry_point='gym_vgdl:VGDLEnv',
    kwargs={
        'game_file': DATA_DIR + 'aliens' + '.txt',
        'map_file': DATA_DIR + 'aliens' + '_lvl0.txt',
        'obs_type': 'features',
        'notable_sprites': ['base', 'bomb', 'alien']
    },
    timestep_limit=1000,
    nondeterministic=True,
)
