from rofunc.learning.RofuncRL.trainers.amp_trainer import AMPTrainer as RofuncAMPTrainer

from .ork_trainer import ORKTrainer
from .ppo_trainer import PPOTrainer
from .sac_trainer import SACTrainer

trainer_map = {
    "PPO": PPOTrainer,
    "SAC": SACTrainer,
    "ORK": ORKTrainer,
    "AMP": RofuncAMPTrainer
}
