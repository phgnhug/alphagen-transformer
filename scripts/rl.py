import json
import os
from typing import Optional, Tuple, List
from datetime import datetime
from pathlib import Path

from openai import OpenAI
import fire

import numpy as np
import torch

from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from alphagen.data.expression import *
from alphagen.data.parser import ExpressionParser
from alphagen.models.linear_alpha_pool import LinearAlphaPool, MseAlphaPool
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils import reseed_everything, get_logger
from alphagen.rl.env.core import AlphaEnvCore

from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import initialize_qlib

from alphagen_llm.client import (
    ChatClient,
    OpenAIClient,
    ChatConfig
)
from alphagen_llm.prompts.system_prompt import EXPLAIN_WITH_TEXT_DESC
from alphagen_llm.prompts.interaction import (
    InterativeSession,
    DefaultInteraction
)


# ============================================================
# AlphaGPT initial pool
# ============================================================

def read_alphagpt_init_pool(seed: int) -> List[Expression]:
    DIR = "./out/llm-tests/interaction"

    parser = build_parser()

    for path in Path(DIR).glob(f"v0_{seed}*"):
        with open(path / "report.json") as f:
            data = json.load(f)

            pool_state = data[-1]["pool_state"]

            return [
                parser.parse(expr)
                for expr, _ in pool_state
            ]

    return []


# ============================================================
# Parser
# ============================================================

def build_parser() -> ExpressionParser:
    return ExpressionParser(
        Operators,
        ignore_case=True,
        non_positive_time_deltas_allowed=False,
        additional_operator_mapping={
            "Max": [Greater],
            "Min": [Less],
            "Delta": [Sub]
        }
    )


# ============================================================
# LLM client
# ============================================================

def build_chat_client(log_dir: str) -> ChatClient:
    logger = get_logger(
        "llm",
        os.path.join(log_dir, "llm.log")
    )

    return OpenAIClient(
        client=OpenAI(
            base_url="https://api.ai.cs.ac.cn/v1"
        ),
        config=ChatConfig(
            system_prompt=EXPLAIN_WITH_TEXT_DESC,
            logger=logger
        )
    )


# ============================================================
# Callback
# ============================================================

class CustomCallback(BaseCallback):

    def __init__(
        self,
        save_path: str,
        test_calculators: List[QLibStockDataCalculator],
        verbose: int = 0,
        chat_session: Optional[InterativeSession] = None,
        llm_every_n_steps: int = 25_000,
        drop_rl_n: int = 5
    ):
        super().__init__(verbose)

        self.save_path = save_path
        self.test_calculators = test_calculators

        os.makedirs(
            self.save_path,
            exist_ok=True
        )

        self.llm_use_count = 0
        self.last_llm_use = 0

        self.obj_history: List[
            Tuple[int, float]
        ] = []

        self.llm_every_n_steps = llm_every_n_steps
        self.chat_session = chat_session
        self._drop_rl_n = drop_rl_n

    # --------------------------------------------------------
    # Step
    # --------------------------------------------------------

    def _on_step(self) -> bool:
        return True

    # --------------------------------------------------------
    # Rollout end
    # --------------------------------------------------------

    def _on_rollout_end(self) -> None:

        if self.chat_session is not None:
            self._try_use_llm()

        self.logger.record(
            "pool/size",
            self.pool.size
        )

        self.logger.record(
            "pool/significant",
            (
                np.abs(
                    self.pool.weights[:self.pool.size]
                ) > 1e-4
            ).sum()
        )

        self.logger.record(
            "pool/best_ic_ret",
            self.pool.best_ic_ret
        )

        self.logger.record(
            "pool/eval_cnt",
            self.pool.eval_cnt
        )

        n_days = sum(
            calculator.data.n_days
            for calculator in self.test_calculators
        )

        ic_test_mean = 0.0
        rank_ic_test_mean = 0.0

        for i, test_calculator in enumerate(
            self.test_calculators,
            start=1
        ):

            ic_test, rank_ic_test = (
                self.pool.test_ensemble(
                    test_calculator
                )
            )

            ic_test_mean += (
                ic_test
                * test_calculator.data.n_days
                / n_days
            )

            rank_ic_test_mean += (
                rank_ic_test
                * test_calculator.data.n_days
                / n_days
            )

            self.logger.record(
                f"test/ic_{i}",
                ic_test
            )

            self.logger.record(
                f"test/rank_ic_{i}",
                rank_ic_test
            )

        self.logger.record(
            "test/ic_mean",
            ic_test_mean
        )

        self.logger.record(
            "test/rank_ic_mean",
            rank_ic_test_mean
        )

        self.save_checkpoint()

    # --------------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------------

    def save_checkpoint(self):

        path = os.path.join(
            self.save_path,
            f"{self.num_timesteps}_steps"
        )

        self.model.save(path)  # type: ignore

        if self.verbose > 1:
            print(
                f"Saving model checkpoint to {path}"
            )

        with open(
            f"{path}_pool.json",
            "w"
        ) as f:

            json.dump(
                self.pool.to_json_dict(),
                f
            )

    # --------------------------------------------------------
    # Show pool
    # --------------------------------------------------------

    def show_pool_state(self):

        state = self.pool.state

        print("---------------------------------------------")

        for i in range(self.pool.size):

            weight = state["weights"][i]
            expr_str = str(state["exprs"][i])
            ic_ret = state["ics_ret"][i]

            print(
                f"> Alpha #{i}: "
                f"{weight}, "
                f"{expr_str}, "
                f"{ic_ret}"
            )

        print(
            f">> Ensemble ic_ret: "
            f"{state['best_ic_ret']}"
        )

        print("---------------------------------------------")

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    def _try_use_llm(self) -> None:

        n_steps = self.num_timesteps

        if (
            n_steps - self.last_llm_use
            < self.llm_every_n_steps
        ):
            return

        self.last_llm_use = n_steps
        self.llm_use_count += 1

        assert self.chat_session is not None

        self.chat_session.client.reset()

        logger = self.chat_session.logger

        logger.debug(
            f"[Step: {n_steps}] "
            f"Trying to invoke LLM "
            f"(#{self.llm_use_count}): "
            f"IC={self.pool.best_ic_ret:.4f}, "
            f"obj={self.pool.best_ic_ret:.4f}"
        )

        try:

            remain_n = max(
                0,
                self.pool.size - self._drop_rl_n
            )

            remain = (
                self.pool.most_significant_indices(
                    remain_n
                )
            )

            self.pool.leave_only(remain)

            self.chat_session.update_pool(
                self.pool
            )

        except Exception as e:

            logger.warning(
                f"LLM invocation failed due to "
                f"{type(e)}: {str(e)}"
            )

    # --------------------------------------------------------
    # Pool property
    # --------------------------------------------------------

    @property
    def pool(self) -> LinearAlphaPool:

        assert isinstance(
            self.env_core.pool,
            LinearAlphaPool
        )

        return self.env_core.pool

    # --------------------------------------------------------
    # Environment core
    # --------------------------------------------------------

    @property
    def env_core(self) -> AlphaEnvCore:

        return (
            self.training_env
            .envs[0]
            .unwrapped
        )


# ============================================================
# Single experiment
# ============================================================

def run_single_experiment(
    seed: int = 0,
    instruments: str = "csi300",
    pool_capacity: int = 10,
    steps: int = 200_000,
    alphagpt_init: bool = False,
    use_llm: bool = False,
    llm_every_n_steps: int = 25_000,
    drop_rl_n: int = 5,
    llm_replace_n: int = 3
):

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    reseed_everything(seed)

    # --------------------------------------------------------
    # Qlib
    # --------------------------------------------------------

    initialize_qlib(
        "~/.qlib/qlib_data/cn_data_2024h1"
    )

    # --------------------------------------------------------
    # LLM replacement
    # --------------------------------------------------------

    llm_replace_n = (
        0
        if not use_llm
        else llm_replace_n
    )

    print(
        f"""
[Main] Starting training process
    Seed: {seed}
    Instruments: {instruments}
    Pool capacity: {pool_capacity}
    Total Iteration Steps: {steps}
    AlphaGPT-Like Init-Only LLM Usage: {alphagpt_init}
    Use LLM: {use_llm}
    Invoke LLM every N steps: {llm_every_n_steps}
    Replace N alphas with LLM: {llm_replace_n}
    Drop N alphas before LLM: {drop_rl_n}
"""
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d%H%M%S"
    )

    tag = (
        "agpt"
        if alphagpt_init
        else
        "rl"
        if not use_llm
        else
        f"llm_d{drop_rl_n}"
    )

    name_prefix = (
        f"{instruments}_"
        f"{pool_capacity}_"
        f"{seed}_"
        f"{timestamp}_"
        f"{tag}"
    )

    save_path = os.path.join(
        "./out/results",
        name_prefix
    )

    os.makedirs(
        save_path,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"[Main] Device: {device}")

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    close = Feature(
        FeatureType.CLOSE
    )

    target = (
        Ref(close, -20)
        / close
        - 1
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    def get_dataset(
        start: str,
        end: str
    ) -> StockData:

        return StockData(
            instrument=instruments,
            start_time=start,
            end_time=end,
            device=device
        )

    segments = [
        (
            "2012-01-01",
            "2019-12-31"
        ),
        (
            "2020-01-01",
            "2020-04-30"
        ),
        (
            "2020-05-01",
            "2020-07-31"
        ),
        (
            "2020-08-01",
            "2020-09-01"
        )
    ]

    datasets = [
        get_dataset(*s)
        for s in segments
    ]

    calculators = [
        QLibStockDataCalculator(
            d,
            target
        )
        for d in datasets
    ]

    # --------------------------------------------------------
    # Pool builder
    # --------------------------------------------------------

    def build_pool(
        exprs: List[Expression]
    ) -> LinearAlphaPool:

        pool = MseAlphaPool(
            capacity=pool_capacity,
            calculator=calculators[0],
            ic_lower_bound=None,
            l1_alpha=5e-3,
            device=device
        )

        if len(exprs) != 0:

            pool.force_load_exprs(
                exprs
            )

        return pool

    # --------------------------------------------------------
    # Initial pool / LLM
    # --------------------------------------------------------

    chat = None
    inter = None
    pool = build_pool([])

    if alphagpt_init:

        pool = build_pool(
            read_alphagpt_init_pool(
                seed
            )
        )

    elif use_llm:

        chat = build_chat_client(
            save_path
        )

        inter = DefaultInteraction(
            build_parser(),
            chat,
            build_pool,
            calculator_train=calculators[0],
            calculators_test=calculators[1:],
            replace_k=llm_replace_n,
            forgetful=True
        )

        pool = inter.run()

    # ========================================================
    # RESUME CHECKPOINT
    # ========================================================

    resume_path = os.environ.get(
        "RESUME_CHECKPOINT"
    )

    if resume_path:

        # ----------------------------------------------
        # Check model checkpoint
        # ----------------------------------------------

        if not os.path.exists(
            resume_path
        ):
            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{resume_path}"
            )

        # ----------------------------------------------
        # Pool checkpoint
        # ----------------------------------------------

        pool_path = (
            resume_path.replace(
                ".zip",
                "_pool.json"
            )
        )

        if not os.path.exists(
            pool_path
        ):
            raise FileNotFoundError(
                f"Pool checkpoint not found: "
                f"{pool_path}"
            )

        print(
            "\n[Resume] Restoring checkpoint"
        )

        print(
            f"[Resume] Model: {resume_path}"
        )

        print(
            f"[Resume] Pool:  {pool_path}"
        )

        # ----------------------------------------------
        # Load pool JSON
        # ----------------------------------------------

        with open(
            pool_path,
            "r"
        ) as f:

            pool_state = json.load(f)

        # ----------------------------------------------
        # Parse expressions
        # ----------------------------------------------

        parser = build_parser()

        exprs = [
            parser.parse(expr)
            for expr in pool_state["exprs"]
        ]

        weights = pool_state["weights"]

        # ----------------------------------------------
        # Rebuild pool
        # ----------------------------------------------

        pool = build_pool([])

        pool.force_load_exprs(
            exprs,
            weights=weights
        )

        print(
            f"[Resume] Restored "
            f"{pool.size} alphas"
        )

        print(
            "[Resume] Pool weights restored"
        )

    # ========================================================
    # Environment
    # ========================================================

    env = AlphaEnv(
        pool=pool,
        device=device,
        print_expr=True
    )

    # ========================================================
    # Callback
    # ========================================================

    checkpoint_callback = CustomCallback(
        save_path=save_path,
        test_calculators=calculators[1:],
        verbose=1,
        chat_session=inter,
        llm_every_n_steps=llm_every_n_steps,
        drop_rl_n=drop_rl_n
    )

    # ========================================================
    # PPO MODEL
    # ========================================================

    if resume_path:

        print(
            "\n[Resume] Loading PPO model..."
        )

        model = MaskablePPO.load(
            resume_path,
            env=env,
            device=device
        )

        print(
            f"[Resume] Resumed model from "
            f"{resume_path}"
        )

        print(
            f"[Resume] Model timestep: "
            f"{model.num_timesteps}"
        )

    else:

        print(
            "\n[Main] Creating new PPO model..."
        )

        model = MaskablePPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(
                features_extractor_class=LSTMSharedNet,
                features_extractor_kwargs=dict(
                    n_layers=2,
                    d_model=128,
                    dropout=0.1,
                    device=device,
                ),
            ),
            gamma=1.,
            ent_coef=0.01,
            batch_size=128,
            tensorboard_log="./out/tensorboard",
            device=device,
            verbose=1,
        )

    # ========================================================
    # TRAIN
    # ========================================================

    print(
        "\n[Main] Starting model.learn()"
    )

    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        tb_log_name=name_prefix,
        reset_num_timesteps=(
            resume_path is None
        ),
    )

    print(
        "\n[Main] Training finished."
    )


# ============================================================
# Main
# ============================================================

def main(
    random_seeds: Union[int, Tuple[int]] = 0,
    pool_capacity: int = 20,
    instruments: str = "csi300",
    alphagpt_init: bool = False,
    use_llm: bool = False,
    drop_rl_n: int = 10,
    steps: Optional[int] = None,
    llm_every_n_steps: int = 25_000
):
    """
    :param random_seeds:
        Random seeds

    :param pool_capacity:
        Maximum size of the alpha pool

    :param instruments:
        Stock subset name

    :param alphagpt_init:
        Use an alpha set pre-generated by LLM
        as the initial pool

    :param use_llm:
        Enable LLM usage

    :param drop_rl_n:
        Drop n worst alphas before invoking LLM

    :param steps:
        Number of additional training steps

    :param llm_every_n_steps:
        Invoke LLM every n steps
    """

    if isinstance(
        random_seeds,
        int
    ):
        random_seeds = (
            random_seeds,
        )

    default_steps = {
        10: 200_000,
        20: 250_000,
        50: 300_000,
        100: 350_000
    }

    for s in random_seeds:

        run_single_experiment(
            seed=s,
            instruments=instruments,
            pool_capacity=pool_capacity,
            steps=(
                default_steps[
                    int(pool_capacity)
                ]
                if steps is None
                else int(steps)
            ),
            alphagpt_init=alphagpt_init,
            drop_rl_n=drop_rl_n,
            use_llm=use_llm,
            llm_every_n_steps=llm_every_n_steps
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    fire.Fire(main)