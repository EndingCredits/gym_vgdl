import gym
from gym import spaces
from .vgdl import core
import pygame
import numpy as np
from .list_space import list_space


class VGDLEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 25
    }

    def __init__(self, game_file = None, map_file = None, obs_type='image', **kwargs):

        # Load game description and level description
        if game_file == None:
            self.game_desc = aliens_game
            self.level_desc = aliens_level
        else:
            with open (game_file, "r") as myfile:
                self.game_desc = myfile.read()
            with open (map_file, "r") as myfile:
                self.level_desc = myfile.read()


        self._obs_type = obs_type
        self.viewer = None
        self.game_args = kwargs
        
        # Need to build a sample level to get the available actions and screensize....
        self.game = core.VGDLParser().parseGame(self.game_desc, **self.game_args)
        self.game.buildLevel(self.level_desc)

        self._action_set = self.game.getPossibleActions()
        self.screen_width, self.screen_height = self.game.screensize

        self.score_last = self.game.score

        # Set action space and observation space

        self.action_space = spaces.Discrete(len(self._action_set))
        if self._obs_type == 'image':
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        elif self._obs_type == 'objects':
            self.observation_space = list_space(spaces.Box(low=-100, high=100, shape=(self.game.lenObservation())))
        elif self._obs_type == 'features':
            self.observation_space = spaces.Box(low=0, high=100, shape=(self.game.lenFeatures()))

        self.screen = pygame.display.set_mode(self.game.screensize, 0, 32)

        self.game.screen = self.screen
        self.game.background = pygame.Surface(self.game.screensize)
        self.game.screen.fill((0, 0, 0))



    def _step(self, a):
        self.game.tick(list(self._action_set.values())[a], True)
        self._update_display()
        state = self._get_obs()
        reward = self.game.score - self.score_last; self.score_last = self.game.score
        terminal = self.game.ended

        return state, reward, terminal, {}


    @property
    def _n_actions(self):
        return len(self._action_set)

    def _update_display(self):
        self.game._drawAll()
        pygame.display.update()

    def _get_image(self):
        self._update_display()
        return np.flipud(np.rot90(pygame.surfarray.array3d(
            pygame.display.get_surface()).astype(np.uint8)))

    def _get_obs(self):
        if self._obs_type == 'image':
            return self._get_image()
        elif self._obs_type == 'objects':
            return self.game.getObservation()
        elif self._obs_type == 'features':
            return self.game.getFeatures()

    def _reset(self):

        # Do things the easy way...
        #del self.game
        #self.game = core.VGDLParser().parseGame(self.game_desc, **self.game_args)
        self.game.reset()
        self.game.buildLevel(self.level_desc)

        self.score_last = self.game.score

        state = self._get_obs()

        return state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            
        
class Padlist(gym.ObservationWrapper):
    def __init__(self, env=None, max_objs=200):
        self.max_objects = max_objs
        super(Padlist, self).__init__(env)
        env_shape = self.observation_space.shape
        env_shape[0] = self.max_objects
        self.observation_space = gym.spaces.Box(low=-100, high=100, shape=env_shape)

    def _observation(self, obs):
        return Padlist.process(obs, self.max_objects)
        
    @staticmethod
    def process(input_list, to_len):
        max_len = to_len
        item_len = len(input_list)
        if item_len < max_len:
          padded = np.pad(np.array(input_list,dtype=np.float32), ((0,max_len-item_len),(0,0)), mode='constant')
          return padded
        else:
          return np.array(input_list, dtype=np.float32)[:max_len]


####################################################################################################


# Example VGDL description text
# The game dynamics are specified as a paragraph of text

aliens_game = """
BasicGame block_size=10
    SpriteSet
        background > Immovable img=oryx/space1 hidden=True
        base    > Immovable    color=WHITE img=oryx/planet
        avatar  > FlakAvatar   stype=sam img=oryx/spaceship1
        missile > Missile
            sam  > orientation=UP    color=BLUE singleton=True img=oryx/bullet2
            bomb > orientation=DOWN  color=RED  speed=0.5 img=oryx/bullet2
        alien   > Bomber       stype=bomb   prob=0.05  cooldown=3 speed=0.8
            alienGreen > img=oryx/alien3
            alienBlue > img=oryx/alien1
        portal  > invisible=True hidden=True
        	portalSlow  > SpawnPoint   stype=alienBlue  cooldown=16   total=20
        	portalFast  > SpawnPoint   stype=alienGreen  cooldown=12   total=20
    
    LevelMapping
        . > background
        0 > background base
        1 > background portalSlow
        2 > background portalFast
        A > background avatar

    TerminationSet
        SpriteCounter      stype=avatar               limit=0 win=False
        MultiSpriteCounter stype1=portal stype2=alien limit=0 win=True
        
        
    InteractionSet
        avatar  EOS  > stepBack
        alien   EOS  > turnAround
        missile EOS  > killSprite

        base bomb > killSprite
        base sam > killSprite scoreChange=1

        base   alien > killSprite
        avatar alien > killSprite scoreChange=-1
        avatar bomb  > killSprite scoreChange=-1
        alien  sam   > killSprite scoreChange=2     
"""

# the (initial) level as a block of characters 
aliens_level = """
1.............................
000...........................
000...........................
..............................
..............................
..............................
..............................
....000......000000.....000...
...00000....00000000...00000..
...0...0....00....00...00000..
................A.............
"""

