import rofunc as rf
from rofunc.learning.utils.utils import set_seed

from simulator.plb.envs import make


def make_env(args, env_cfg):
    env_name = '{}-{}-v1'.format(args.task, args.mode)
    set_seed(env_cfg.env.seed)

    env = make(env_name, sdf_loss=env_cfg.env.sdf_loss, density_loss=env_cfg.env.density_loss,
                     contact_loss=env_cfg.env.contact_loss, soft_contact_loss=env_cfg.env.soft_contact_loss)

    rf.logger.beauty_print('Type of env: {}'.format(type(env.env)), type='info')
    return env
