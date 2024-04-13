


import logging
from typing import List, Tuple
import textattack
from textattack.shared.utils import logger
from textattack import AttackArgs
from textattack.attack_results import (
    FailedAttackResult,
    MaximizedAttackResult,
    SkippedAttackResult,
    SuccessfulAttackResult,
)
from textattack.transformations import Transformation
import torch
import tqdm
from dtcover.coverage.coverage_infer import get_covered_neurons
from dtcover.injection import get_injection_module
from dtcover.llm_attacker import LLMAttacker
from dtcover.metrics import NeuralMetric


def _attack(attacker: LLMAttacker, metrics: List[Tuple[NeuralMetric, List[Transformation]]], coverage_strategy="all", dtcov_logger=None):
    """Internal method that carries out attack.

    No parallel processing is involved.
    coverage_strategy: all, diff, success, failure
    """
    # if torch.cuda.is_available():
    #     self.attack.cuda_()

    if attacker._checkpoint:
        num_remaining_attacks = attacker._checkpoint.num_remaining_attacks
        worklist = attacker._checkpoint.worklist
        worklist_candidates = attacker._checkpoint.worklist_candidates
        logger.info(
            f"Recovered from checkpoint previously saved at {attacker._checkpoint.datetime}."
        )
    else:
        if attacker.attack_args.num_successful_examples:
            num_remaining_attacks = attacker.attack_args.num_successful_examples
            # We make `worklist` deque (linked-list) for easy pop and append.
            # Candidates are other samples we can attack if we need more samples.
            worklist, worklist_candidates = attacker._get_worklist(
                attacker.attack_args.num_examples_offset,
                len(attacker.dataset),
                attacker.attack_args.num_successful_examples,
                attacker.attack_args.shuffle,
            )
        else:
            num_remaining_attacks = attacker.attack_args.num_examples
            # We make `worklist` deque (linked-list) for easy pop and append.
            # Candidates are other samples we can attack if we need more samples.
            worklist, worklist_candidates = attacker._get_worklist(
                attacker.attack_args.num_examples_offset,
                len(attacker.dataset),
                attacker.attack_args.num_examples,
                attacker.attack_args.shuffle,
            )

    if not attacker.attack_args.silent:
        print(attacker.attack, "\n")

    pbar = tqdm.tqdm(total=num_remaining_attacks, smoothing=0, dynamic_ncols=True)
    if attacker._checkpoint:
        num_results = attacker._checkpoint.results_count
        num_failures = attacker._checkpoint.num_failed_attacks
        num_skipped = attacker._checkpoint.num_skipped_attacks
        num_successes = attacker._checkpoint.num_successful_attacks
    else:
        num_results = 0
        num_failures = 0
        num_skipped = 0
        num_successes = 0

    sample_exhaustion_warned = False
    # Total Coverage States
    while worklist:
        idx = worklist.popleft()
        try:
            example, ground_truth_output = attacker.dataset[idx]
        except IndexError:
            continue
        example = textattack.shared.AttackedText(example)
        if attacker.dataset.label_names is not None:
            example.attack_attrs["label_names"] = attacker.dataset.label_names
        try:
            # Attack !!!!!!!
            result = attacker.attack.attack(example, ground_truth_output)

            for metric, trans_list in metrics:
                model_config = attacker.attack.goal_function.model.model.config
                module_re = get_injection_module(model_config.model_type, metric.kind)
                # Get original_result coverage
                original_cover_neurons = get_covered_neurons(
                    attacker.attack.goal_function.model,
                    text=example.text,
                    metric=metric,
                    metric_name=type(metric).__name__,
                    module_name=module_re
                )

                # print coverage
                metric.state = original_cover_neurons.states
                original_cover_value = metric.compute().item()
                if dtcov_logger is not None:
                    dtcov_logger.info(f"[original]:[sample_coverage]:[{type(metric).__name__}]:[{idx}]:{original_cover_value}")
                metric.reset()
                
                # Get perturbed_result coverage
                perturbed_cover_neurons = get_covered_neurons(
                    attacker.attack.goal_function.model,
                    text=result.perturbed_result.attacked_text.text,
                    metric=metric,
                    metric_name=type(metric).__name__,
                    module_name=module_re
                )
                # print coverage
                metric.state = perturbed_cover_neurons.states
                perturbed_cover_value = metric.compute().item()
                if dtcov_logger is not None:
                    dtcov_logger.info(f"[perturbed]:[sample_coverage]:[{type(metric).__name__}]:[{idx}]:{perturbed_cover_value}")
                metric.reset()

                # Merge coverage
                if coverage_strategy == "all":
                    for transformation in trans_list:
                        if transformation.cover_neurons_states is None:
                            transformation.cover_neurons_states = perturbed_cover_neurons
                        else:
                            transformation.cover_neurons_states.update_coverage(perturbed_cover_neurons)
                elif coverage_strategy == "diff":
                    """
                    This strategy will be made public after acceptance.
                    """
                else:
                    raise Exception("Unsupported coverage strategy")

                # print merged coverage
                for transformation in trans_list:
                    if transformation.cover_neurons_states is None:
                        cover_value = 0.0
                    else:
                        metric.state = transformation.cover_neurons_states.states
                        cover_value = metric.compute().item()
                        metric.reset()
                    if dtcov_logger is not None:
                        dtcov_logger.info(f"[total]:[coverage]:[{type(metric).__name__}/{type(transformation).__name__}]:[{idx}]:{cover_value}")
                    
        except Exception as e:
            raise e
            # traceback.print_exc()
            # continue
        if (
            isinstance(result, SkippedAttackResult) and attacker.attack_args.attack_n
        ) or (
            not isinstance(result, SuccessfulAttackResult)
            and attacker.attack_args.num_successful_examples
        ):
            if worklist_candidates:
                next_sample = worklist_candidates.popleft()
                worklist.append(next_sample)
            else:
                if not sample_exhaustion_warned:
                    logger.warn("Ran out of samples to attack!")
                    sample_exhaustion_warned = True
        else:
            pbar.update(1)

        attacker.attack_log_manager.log_result(result)
        if not attacker.attack_args.disable_stdout and not attacker.attack_args.silent:
            print("\n")
        num_results += 1

        if isinstance(result, SkippedAttackResult):
            num_skipped += 1
        if isinstance(result, (SuccessfulAttackResult, MaximizedAttackResult)):
            num_successes += 1
        if isinstance(result, FailedAttackResult):
            num_failures += 1
        pbar.set_description(
            f"[Succeeded / Failed / Skipped / Total] {num_successes} / {num_failures} / {num_skipped} / {num_results}"
        )

        if (
            attacker.attack_args.checkpoint_interval
            and len(attacker.attack_log_manager.results)
            % attacker.attack_args.checkpoint_interval
            == 0
        ):
            new_checkpoint = textattack.shared.AttackCheckpoint(
                attacker.attack_args,
                attacker.attack_log_manager,
                worklist,
                worklist_candidates,
            )
            new_checkpoint.save()
            attacker.attack_log_manager.flush()

    pbar.close()
    print()
    # Enable summary stdout
    if not attacker.attack_args.silent and attacker.attack_args.disable_stdout:
        attacker.attack_log_manager.enable_stdout()

    if attacker.attack_args.enable_advance_metrics:
        attacker.attack_log_manager.enable_advance_metrics = True

    attacker.attack_log_manager.log_summary()
    attacker.attack_log_manager.flush()
    print()


def attack_dataset(attacker: LLMAttacker, metrics: List[Tuple[NeuralMetric, List[Transformation]]], coverage_strategy="all", dtcov_logger=None):
    """Attack the dataset.

    Returns:
        :obj:`list[AttackResult]` - List of :class:`~textattack.attack_results.AttackResult` obtained after attacking the given dataset..
    """
    if attacker.attack_args.silent:
        logger.setLevel(logging.ERROR)

    if attacker.attack_args.query_budget:
        attacker.attack.goal_function.query_budget = attacker.attack_args.query_budget

    if not attacker.attack_log_manager:
        attacker.attack_log_manager = AttackArgs.create_loggers_from_args(
            attacker.attack_args
        )

    textattack.shared.utils.set_seed(attacker.attack_args.random_seed)
    if attacker.dataset.shuffled and attacker.attack_args.checkpoint_interval:
        # Not allowed b/c we cannot recover order of shuffled data
        raise ValueError(
            "Cannot use `--checkpoint-interval` with dataset that has been internally shuffled."
        )

    attacker.attack_args.num_examples = (
        len(attacker.dataset)
        if attacker.attack_args.num_examples == -1
        else attacker.attack_args.num_examples
    )
    if attacker.attack_args.parallel:
        raise Exception("Parallel Attack with coverage is not supported now.")
    else:
        _attack(attacker, metrics, coverage_strategy, dtcov_logger)

    if attacker.attack_args.silent:
        logger.setLevel(logging.INFO)

    return attacker.attack_log_manager.results
