import os
import datetime
import itertools
import getpass

import argparse
from glob import glob
import os
import random

user_folders = {
    'output-dirname': {
        'arorakus': '/home/mila/a/arorakus/scratch/mgs_mt',
    },
    'base-dir': {
        'arorakus': '/home/mila/a/arorakus/wdir/mgs/mgs_mt',
    }
}

current_user = getpass.getuser()

parser = argparse.ArgumentParser()
parser.add_argument('--nhrs', type=int, default=24)
parser.add_argument('--ngpus', type=int, default=1)
parser.add_argument('--ncpus', type=int, default=4)
parser.add_argument('--dryrun', action='store_true')
parser.add_argument('--print-commands', action='store_true')
args = parser.parse_args()

# Global extras
extra_flags = []  # extra flags, specified as {--flagname: None} passed to all jobs
extra_args = []  # extra {--argname: argvalue} args passed to all jobs
extra_name = ''  # extra name suffix appended to `expr`
eval_mode = False # set this to True when want to skip save-base-dir


DEFAULT_COMMON_SETTINGS = {
    'script': "fairseq-train data-bin/iwslt14.tokenized.de-en",
    '--no-epoch-checkpoints': None,
    '--arch': 'transformer_iwslt_de_en',
    '--share-decoder-input-output-embed': None,
    '--optimizer': 'adam',
    '--clip-norm': '1.0',
    '--lr': '5e-4',
    '--lr-scheduler': 'inverse_sqrt',
    '--warmup-updates': '4000',
    '--dropout': '0.3',
    '--weight-decay': '0.0001',
    '--criterion': 'label_smoothed_cross_entropy',
    '--label-smoothing': '0.1',
    '--max-tokens': '4096',
    '--eval-bleu': None,
    '--eval-bleu-detok moses': None,
    '--eval-bleu-remove-bpe': None,
    '--eval-bleu-print-samples': None,
    '--best-checkpoint-metric': 'bleu',
    '--maximize-best-checkpoint-metric': None,
    '--no-progress-bar': None,
    '--patience': '-1',
    '--log-format': 'simple',
    '--ddp-backend': 'legacy_ddp',
}

name_fields = []

# ======= Experiments
DEFAULT_GGS_SETTINGS = {
    '--task': 'translation_ggs',
}
for k, v in DEFAULT_COMMON_SETTINGS.items():
    if k not in DEFAULT_GGS_SETTINGS:
        DEFAULT_GGS_SETTINGS[k] = v

# -- MLE pretrain / baseline
if False:
    GRID = True
    common_settings = DEFAULT_COMMON_SETTINGS
    grids = [
        {
            '--task': ['translation'],
        },
    ]
    expr = 'iwslt_mle'


# -- GGS tune
if True:
    GRID = True
    common_settings = DEFAULT_GGS_SETTINGS
    common_settings['--restore-file'] = 'iwslt_mle/checkpoint_best.pt'
    common_settings['--clip-norm'] = '1.0'

    common_settings['--reset-optimizer'] = None
    common_settings['--reset-meters'] = None
    common_settings['--reset-lr-scheduler'] = None
    common_settings['--max-tokens'] = '16384'
    common_settings['--lr-scheduler'] = 'fixed'
    common_settings['--lr-shrink'] = '0.9'
    common_settings['--warmup-updates'] = '0'
    common_settings['--lr'] = '6.25e-5'
    common_settings['--update-freq'] = '4'
    common_settings['--max-tokens'] = '16384'
    common_settings['--user-dir'] = 'ggs'

    grids = [
        {
            '--ggs-num-samples': ['4'],
            '--ggs-metric': ['sentence_bleu'],
            '--ggs-beta': ['100.0'],
            '--ggs-noise': ['1.0'],
            '--noise-scaling': ['uniform-global']
        },
    ]
    expr = 'iwslt_ggs_tune'


# -- GGS from scratch
if False:
    GRID = True
    common_settings = DEFAULT_GGS_SETTINGS
    common_settings['--clip-norm'] = '1.0'
    common_settings['--max-tokens'] = '4096'
    common_settings['--update-freq'] = '1'

    grids = [
        {
            '--ggs-num-samples': ['4'],
            '--noise-scaling': ['uniform-global'],
            '--ggs-metric': ['sentence_bleu'],
            '--ggs-noise': ['1.0', '0.1', '0.01'],
            '--ggs-beta': ['1.0', '10.0', '100.0'],
        },
    ]
    expr = 'iwslt_ggs_fromscratch'




# ========= Run the job combinations (you shouldn't need to modify/read this code; keeping everything in one file.)
# Setup the base output directory of the form:
#       {args.output_base_dir}/{expr}{extra_name}/{datetime}
# E.g.
#       project_x/output/expr1/0113_0330
now = datetime.datetime.now()
datetime = now.strftime("%m%d_%H%M")
output_dir = os.path.join(user_folders['output-dirname'][current_user], "%s%s" % (expr, extra_name), datetime)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print("Output Directory: %s" % output_dir)

if GRID:
    # Make combinations of grid values; each combination is a 'job'
    jobs = []
    for grid in grids:
        individual_options = [[{k: v} for v in values]
                              for k, values in grid.items()]
        product_options = list(itertools.product(*individual_options))
        jobs += [{k: v for d in option_set for k, v in d.items()}
                 for option_set in product_options]

    merged_grid = {}
    for grid in grids:
        for key in grid:
            merged_grid[key] = [] if key not in merged_grid else merged_grid[key]
            merged_grid[key] += grid[key]
    name_fields = {key for key in merged_grid if len(set(merged_grid[key])) > 1}

if args.dryrun:
    print("NOT starting {} jobs:".format(len(jobs)))
else:
    print("Starting {} jobs:".format(len(jobs)))

# Do the runs
for job in jobs:
    # Make the name
    name = '%s%s' % (extra_name, job.get('name', expr))
    if len(name_fields) > 0:
        name += '__'
        for k in name_fields:
            name += '__'
            if '--%s' % k in job:
                name += '%s=%s' % (k, str(job['--%s' % k]))
            elif '-%s' % k in job:
                name += '%s=%s' % (k, str(job['-%s' % k]))
            elif k in job:
                if k.startswith('--'):
                    k_ = k[2:]
                elif k.startswith('-'):
                    k_ = k[1:]
                else:
                    k_ = k
                if 'restore-file' in k:
                    opt_val = job[k].split('/')[-2]
                else:
                    opt_val = job[k]
                name += '%s=%s' % (k_, str(opt_val))

    if '/' in name:
        name = name.replace('/', '_slash_')
    print('\n' + name)

    # Pass the name and output directory to the downstream python command.
    if eval_mode is False:
        job['--save-dir'] = os.path.join(output_dir, name)
        job['--tensorboard-logdir'] = os.path.join(output_dir, name)
        os.makedirs(job['--save-dir'])

    # Make the python command
    script = common_settings.get('script', job.get('script'))
    cmd = [script]

    for arg, val in common_settings.items():
        if isinstance(val, list):
            cmd.append(arg)
            for item in val:
                cmd.append(item)
        else:
            arg_, val_ = str(arg), str(val)
            if arg_ == 'name' or arg_ == 'script' or arg_ in job:
                continue
            cmd.append(arg_)
            if val is not None:
                cmd.append(val_)

    for arg, val in job.items():
        arg_, val_ = str(arg), str(val)
        if arg_ == 'name' or arg_ == 'script':
            continue
        cmd.append(arg_)
        if val is not None:
            cmd.append(val_)

    for arg, val in extra_args:
        arg_, val_ = str(arg), str(val)
        cmd.append(arg_)
        if val is not None:
            cmd.append(val_)

    for flag in extra_flags:
        flag = str(flag)
        cmd.append(flag)

    cmd = ' '.join(cmd)
    if args.print_commands:
        print(cmd)

    # Make a {name}.slurm file in the {output_dir} which defines this job.
    slurm_script_path = os.path.join(output_dir, '%s.slurm' % name)
    slurm_command = "sbatch %s" % slurm_script_path

    # Make the .slurm file
    with open(slurm_script_path, 'w') as slurmfile:
        slurmfile.write("#!/bin/bash\n")
        slurmfile.write("#SBATCH --job-name" + "=" + name + "\n")
        slurmfile.write("#SBATCH --open-mode=append\n")
        slurmfile.write("#SBATCH --output=%s.out\n" % (os.path.join(output_dir, name)))
        slurmfile.write("#SBATCH --error=%s.err\n" % (os.path.join(output_dir, name)))
        slurmfile.write("#SBATCH --export=ALL\n")
        slurmfile.write("#SBATCH --time=%d:00:00\n" % args.nhrs)
        slurmfile.write("#SBATCH --mem=50G\n")
        slurmfile.write("#SBATCH --gres=gpu:%d\n" % args.ngpus)
        # slurmfile.write("#SBATCH --partition=p40_4,p100_4,v100_sxm2_4")
        slurmfile.write("#SBATCH -c %d\n" % args.ncpus)
        slurmfile.write('#SBATCH --signal=USR1@60\n')
        slurmfile.write('term_handler () {\n\
    # catch and ignore TERM. we get multiple terms during shutdown, so best\n\
    # to just do nothing\n\
    # but still keep going with the python process\n\
    wait "$CHILD"\n\
}\n\n')
        slurmfile.write('usr1_handler () {\n\
    echo "SLURM signaling preemption/times up (SLURM_PROCID $SLURM_PROCID)." \n\
    kill -s INT "$CHILD"  # send ctrl-c to python\n\
    if {SHOULD_REQUEUE} && [ "$SLURM_PROCID" -eq "0" ]; then\n\
        echo "Waiting 5s and resubmitting..."\n\
        sleep 5\n\
        echo "Resubmitting..."\n\
        scontrol requeue $SLURM_JOB_ID\n\
    fi\n\
    wait "$CHILD"\n\
}\n\n')
        slurmfile.write("trap 'usr1_handler' USR1\n")
        slurmfile.write("trap 'term_handler' TERM\n")
        slurmfile.write("cd " + user_folders['base-dir'][current_user] + '\n')
        slurmfile.write("srun " + cmd)
        slurmfile.write("\n")

    if not args.dryrun:
        os.system("%s &" % slurm_command)

    print("Follow logfile: tail -f %s" % (os.path.join(output_dir, name + '.out')))
