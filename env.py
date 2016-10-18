
from gym.envs.atari import AtariEnv
import numpy as np
from scipy.misc import imresize

class Atari(AtariEnv):
    """Like atari env, but does not fuck up when atari shows sprites on every second/third frame (optimization).
    To observe the difference between the two environments, launch "alien" or air raid games and render a few frames in a sequence.
    In vanilla rnn, you'll notice flickering effect."""
    def __init__(self,game='pong',frameskip=2,
                 image_size=(210, 160),
                 interpolation='lanczos',
                 grayscale=False,
                 deflicker = lambda frames: np.max(frames,axis=0),
                 deflicker_buffer_size=2):
        self.image_size= image_size
        self.interpolation = interpolation
        self.grayscale = grayscale
        self.deflicker = deflicker
        buffer_shape = [deflicker_buffer_size]+list(image_size)+[1 if grayscale else 3]
        self.images_buffer = np.zeros(buffer_shape,dtype='uint8')

        super(Atari,self).__init__(game=game, obs_type='image', frameskip=frameskip, repeat_action_probability=0.)
    def _step(self, a):
        assert self._obs_type == 'image'
        reward = 0.0
        action = self._action_set[a]

        if isinstance(self.frameskip, int):
            num_steps = self.frameskip
        else:
            num_steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])
        for _ in range(num_steps):
            reward += self.ale.act(action)
            self._get_obs()  # need to call this guy to then correctly remove flickering

        ob = self.get_observation()

        return ob, reward, self.ale.game_over(), {}
    def get_observation(self):
        """
        :return: observation with rescaling and deflickering applied
        """
        return self.deflicker(self.images_buffer)
    def _get_image(self):
        raw_img = super(Atari,self)._get_image()
        img = imresize(raw_img,self.image_size,interp=self.interpolation)

        if self.grayscale:
            raw_img = img.mean(0,keepdims=True)
        self.images_buffer = np.concatenate([self.images_buffer[1:],img[None,...]],axis=0)

        return raw_img


    def _reset(self):
        self.images_buffer = np.zeros_like(self.images_buffer,dtype=self.images_buffer.dtype)
        super(Atari,self)._reset()
        return self.get_observation()