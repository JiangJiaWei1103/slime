"""Run slime with IsoExec through the external integration entrypoint."""

from train import train

from isoexec.integrations.slime.actor import IsoExecMegatronTrainRayActor
from isoexec.integrations.slime.config import prepare_args
from slime.utils.arguments import parse_args


if __name__ == "__main__":
    args, restore_plan = parse_args(return_restore_plan=True)
    prepare_args(args)
    train(args, restore_plan, actor_cls=IsoExecMegatronTrainRayActor)
