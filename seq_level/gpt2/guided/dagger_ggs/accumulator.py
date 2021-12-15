from copy import deepcopy
from timeit import default_timer as timer

import logging
import numpy as np
import os
import queue as q

import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from seq_level.gpt2.guided.dagger_ggs.ggs_efficient_utils import (InstanceType, 
                                                                  get_dataloader)
import seq_level.gpt2.guided.utils as ggs_utils

timer_context = ggs_utils.TimerContext()

def cleanup():
    dist.destroy_process_group()

def multigpu_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def start_scorer_training_data_accumulation(buffer, dataset, model, score_model, tokenizer, args):
    # torch.multiprocessing.set_sharing_strategy('file_system')
    mp.set_start_method('spawn')

    world_size = torch.cuda.device_count()
    ctx = mp.get_context('spawn')

    queue = ctx.Queue(100000)
    event = ctx.Event()
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=accumulate_scorer_training_data_v2, 
                        args=(rank, dataset, model, score_model, 
                              tokenizer, args, queue, event, world_size))
        processes.append(p)
        p.start()

    # ctxt = mp.spawn(accumulate_scorer_training_data_v2, 
    #                 args=(dataset, model, score_model, 
    #                          tokenizer, args, queue, event, world_size),
    #                 nprocs=world_size,
    #                 join=False)

    is_done = [False] * world_size
    while not np.all(is_done):
        try:
            data = queue.get(block=True)
            print(f"Queue Size: {queue.qsize()}")
            if len(data) == 2:
                is_done[data[0]] = True
            else:
                buffer.append(*data)
                del data
        except Exception as e:
            print()
            print("Error Queue:" + str(e))
            print("Sleeping for 100")
            time.sleep(100)

        
    print("Setting Event")
    event.set()
    print("Waiting for Spawn to join.")

    # ctxt.join(timeout=100)
    for idx, p in enumerate(processes):
        print(f"Waiting for Process: {idx}")
        p.join(timeout=1000)
        print(f"Joined Process: {idx}")


def accumulate_scorer_training_data_v2(device, dataset, model, score_model, tokenizer, args, queue, event, world_size=1):
    multigpu_setup(device, world_size)

    train_dataloader = get_dataloader(args, dataset, device, world_size, batch_size=1)
    train_dataloader.sampler.set_epoch(0)

    model = model.to(device)
    score_model = score_model.to(device)

    total_num_batches = len(train_dataloader)
    for step, (batch_id, batch) in enumerate(train_dataloader):
        accumulate_scorer_training_data(step, batch_id[0], batch[0], model, 
                                        score_model, tokenizer, args, device, queue=queue)

        if device == 0 and step % args.print_every == 0:
            scorer_acc_timer = timer_context.get_timer('scorer_data_acc_time')
            print(f"Rank: {device}:: Aggregated Batches:  {step}/{total_num_batches}. " + \
                        f"Avg step time: {scorer_acc_timer.avg_time():.3f}")

    if device == 0:
        print()
    scorer_acc_timer = timer_context.get_timer('scorer_data_acc_time')
    print(f"Rank: {device}:: Aggregated: {total_num_batches} items in " + \
            f"{scorer_acc_timer.cuml_time():.3f} seconds.")

    queue.put((device, "Done"))
    while (not queue.empty()):
        print(f"Rank: {device}:: Process done! but queue ({queue.qsize()}) is not empty. Sleeping for 30 secs.")
        time.sleep(30)
        continue

    print(f"Rank: {device}:: Waiting for event.")
    event.wait()
    print(f"Rank: {device}:: Event received.")

    cleanup()




def accumulate_scorer_training_data(step, batch_id, batch, model, 
                                      score_model, tokenizer, args, device, queue=None):
    """ This method does a forward pass over the original model and 
        the perturbed model to compute the yo_i, the decoded output corresponding
        to the input x using the original model, and yp_i, the decoding output corresponding
        to the perturbed model. 
        The perturbations are sampled from $\Deta \sim Q_{MGS}$.

        It returns a set of tuples with each tuple of the form (x_i, y_i, yo_i, yp_i, \Delta).
    """
    data = {}
    with timer_context('scorer_data_acc_time') as ctxt_timer:
        start_time = timer()
        model = model.to(device=device)
        score_model = score_model.to(device)
        batch = batch.to(device=device)

        if batch.size(1) < args.context_length + 1:
            logging.error(
                f"Batch at step: {step} has sequences ({batch.size(1)})" + \
                    f"shorter than the context length ({args.context_length})")
            return None

        inp, target = batch[:, :-1], batch[:, 1:]
        max_length = ggs_utils.max_length(target, tokenizer.eos_token_id, args)

        _, cur_decodings, cur_distances = ggs_utils.decode_and_distance(model, 
                                                tokenizer, batch, score_model, max_length, device, args, average_distance=False)

        idx = f'accum_{batch_id}'
        data['idx'] = idx
        data['non_pert'] = (idx, InstanceType.NON_PERTURBED, batch_id, batch.clone(),
                            deepcopy(model), cur_decodings.clone(), cur_distances.clone())
      
        if queue is not None:
            queue.put(data['non_pert'])

        # Get the current MLE gradients
        model_ = deepcopy(model)
        model_ = model_.to(device)
        inp, target = batch[:, :-1], batch[:, 1:]
        model_.eval()
        model_with_grad, _ = ggs_utils.mle_grad(model_, 
                                inp, target, args.pad_token_id, 
                                args.max_grad_norm)
        model_with_grad = model_with_grad.to(device)

        perturbed_models, log_rhos, noise_magnitudes, rng_states, perturb_types = \
                ggs_utils.perturb(model, model_with_grad, args.ggs_num_samples, 
                            args.ggs_noise, noise_scale=args.noise_scale,
                            zero_dist_only=args.zero_dist_only,
                            mle_dist_only=args.mle_dist_only,
                            include_mle_gradient=args.include_mle_gradient)

        data['pert'] = []
        for i, (p_model, log_rho, noise_mag, rng_state, perturb_type) in \
                    enumerate(zip(perturbed_models, log_rhos,                                      
                            noise_magnitudes, rng_states, perturb_types)):
            p_model = p_model.to(device)
            _, per_decodings, per_distances = ggs_utils.decode_and_distance(p_model,
                                                  tokenizer, batch, score_model, max_length,
                                                    device, args, average_distance=False)

            data['pert'].append((idx, InstanceType.PERTURBED, batch_id, batch.clone(), deepcopy(model), 
                                    per_decodings.clone(), per_distances.clone(), log_rho, noise_mag,
                                     rng_state, perturb_type))

            if queue is not None:
                queue.put(data['pert'][-1])

        end_time = timer()
        ctxt_timer.timeit(start_time, end_time)
    return data


def add_args(parser):
    parser.add_argument(
        "--aggregated-data-size", type=int, default=2000,
    )
    parser.add_argument(
        "--aggregated-data-path", type=str,
    )
    parser.add_argument(
        "--save-aggregated-data", action='store_true',
    )
    parser.add_argument(
        "--use-saved-aggregated-data", action='store_true',
    )
    return parser