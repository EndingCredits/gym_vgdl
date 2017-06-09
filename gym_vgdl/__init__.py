from gym.envs.registration import registry, register, make, spec
from gym_vgdl.vgdl_env import VGDLEnv
import os

DATA_DIR = os.path.dirname(__file__) + '/vgdl/games/' 

classes = {
    'aliens': ['alien', 'base', 'bomb', 'sam'],
    'boulderdash': ['boulder', 'butterfly', 'crab', 'diamond', 'exitdoor', 'wall'],
    'chase': ['angry', 'carcass', 'scared', 'wall'],
    'frogs': ['goal', 'log', 'truck', 'wall'],
    'missilecommand': ['city', 'explosion', 'incoming'],
    'portals': [ 'goal', 'portalentry', 'portalexit', 'random', 'straight', 'wall' ],
    'survivezombies': [ 'bee', 'flower', 'hell', 'honey', 'zombie'],
    'zelda': ['enemy', 'goal', 'key', 'wall']
}

resources = {
    'aliens': [],
    'boulderdash': ['diamond'],
    'chase': [],
    'frogs': [],
    'missilecommand': [],
    'portals': [],
    'survivezombies': ['honey'],
    'zelda': []
}

for game in ['aliens', 'boulderdash', 'chase', 'frogs', 'missilecommand', 'portals', 'survivezombies', 'zelda']:
    register(
        id='vgdl_{}-v0'.format(game),
        entry_point='gym_vgdl:VGDLEnv',
        kwargs={
            'game_file': DATA_DIR + game + '.txt',
            'map_file': DATA_DIR + game + '_lvl0.txt',
            'obs_type': 'image',
            'block_size': 5
        },
        timestep_limit=1000,
        nondeterministic=True,
    )


    register(
        id='vgdl_{}_objects-v0'.format(game),
        entry_point='gym_vgdl:VGDLEnv',
        kwargs={
            'game_file': DATA_DIR + game + '.txt',
            'map_file': DATA_DIR + game + '_lvl0.txt',
            'obs_type': 'objects',
            'notable_sprites': classes[game],
            'notable_resources': resources[game],
            'block_size': 10
        },
        timestep_limit=1000,
        nondeterministic=True,
    )

    register(
        id='vgdl_{}_features-v0'.format(game),
        entry_point='gym_vgdl:VGDLEnv',
        kwargs={
            'game_file': DATA_DIR + game + '.txt',
            'map_file': DATA_DIR + game + '_lvl0.txt',
            'obs_type': 'features',
            'notable_sprites': classes[game],
            'notable_resources': resources[game],
            'block_size': 10
        },
        timestep_limit=1000,
        nondeterministic=True,
    )
