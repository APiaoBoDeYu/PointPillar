import _init_path

import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt

import torch
#distributed是多进程的，会分配n个进程对应n块gpu，而DataParallel是单进程控制的，所以存在着PIL（全局解释器锁）的问题。
#distributed在每个进程内都维护了一个optimizer，每个进程都能够独立完成梯度下降这一优化步骤。因此，在各进程梯度计算完成后，只需要将梯度进行汇总平均，再由主进程将其广播到所有进程，之后各进程就可以用该梯度来更新参数了。因为各进程上的模型初始化相同，更新模型参数时所用的梯度相同，所以各进程上的模型参数始终保持相同。
#而DataParallel全局只维护了一个optimizer，只有一个进程能执行梯度下降。因此，DataParallel在主进程上进行梯度汇总和参数更新后，需要将更新后的模型参数广播到其他gpu上，所以传输的数据量大大增加，训练速度大大降低。
import torch.distributed as dist # 多卡训练

import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model


def parse_config():
    """
    解析命令行参数，配置文件和路径
    """
    #建立解析对象
    parser = argparse.ArgumentParser(description='arg parser')
    #增加属性：给xx实例增加一个aa属性 # xx.add_argument("aa")
    parser.add_argument('--cfg_file', type=str, default="cfgs/custom_models/custom_pointpillar.yaml", help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')#?
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')#?
    parser.add_argument('--fix_random_seed', action='store_true', default=True, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    #--loacl_rank”是必须声明的，但它不是由用户填写的，而是由pytorch为用户填写，也就是说这个值是会被自动赋值为当前进程在本机上的rank
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    #属性给与args实例： 把parser中设置的所有"add_argument"给返回到args子类实例当中， 那么parser中增加的属性内容都会在args实例中，使用即可。
    args = parser.parse_args()
    #把配置文件中的参数读取到cfg中
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem # 最后一个路径组件，除去后缀 eg:custom_pointpillar
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml' --> custom_models

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg) # 通过list设置config
    #把你命令行传的参数args和配置文件传的参数返回
    return args, cfg


def main():
    args, cfg = parse_config()
    # 单GPU训练
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        #多GPU训练,nccl是底层的分布式训练库
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        ) # 调用common_utils中的init_dist_pytorch方法
        dist_train = True
    #在传参和配置文件中选取一个batch size，注意区分是单个GPU一个batch的大小还是，所有GPU一个batch的大小
    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU # batch_size: 4
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus # 根据GPU数量计算batch_size

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs # epochs: 80

    if args.fix_random_seed:
        common_utils.set_random_seed(666) # 设定随机种子，使得随机数具有可重复性
    # 输出的文件夹
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    # 模型输出的文件夹 
    ckpt_dir = output_dir / 'ckpt'
    # 创建文件夹
    # parents：如果父目录不存在，是否创建父目录
    # exist_ok：只有在目录不存在时创建目录，目录已存在时不会抛出异常
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # /home/ggj/ObjectDetection/OpenPCDet/output/kitti_models/pointpillar/default/log_train_20211028-093433.txt
    # 日志文件的路径
    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    # 创建日志记录器
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    # 如果没设置GPU就用所有的GPU：‘ALL’
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    # 如果是多卡并行训练，记录总的total_batch_size
    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    # 一行行记录我们传入的参数
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    # 将配置文件记录到log文件中
    log_config_to_file(cfg, logger=logger)
    # 如果单GPU训练，复制配置文件
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir)) # os.system调用shell命令

    # 初始化tensorboard,用于模型和训练过程可视化，只在主进程中进行
    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # -----------------------create dataloader & network & optimizer---------------------------
    # 1.构建dataset, dataloader, sampler
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs
    )

    # 2.构建网络，只是搭建模型结构，（只是包含了各个模块）
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    # 如果设置了BN同步则进行同步设置
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # 将模型放到GPU上
    model.cuda()

    # 3.构建优化器
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # 4.如果可能，尽量加载之前的模型权重
    start_epoch = it = 0 # 起始epoch
    last_epoch = -1 # 上一次的epoch
    # 如果存在预训练模型则加载模型参数
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)
    # 如果存在断点训练，则加载之前训练的权重，包括优化器
    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        # 如果没有写权重位置，也会取权重文件夹，查找是否存在之前训练的权重
        # 如果存在，则加载最后一次的权重文件
        ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            it, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
            )
            last_epoch = start_epoch + 1

    # 5.设置模型为训练模式
    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        #https://blog.csdn.net/weixin_41978699/article/details/121412128
        #模型放到多个GPU
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    # 将模型结构记录到日志中

    logger.info(model)

    # 6.构建调度器
    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag)) # kitti_models/pointpillar(default)
    # 调用train_utils中的train_model函数
    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,#隔多大间隙，保存一次
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch
    )

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    # 训练结束后，对模型进行评估
    # 1.构建test数据集和加载器
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )
    # 2.构造评估结果输出文件夹
    # /home/ggj/ObjectDetection/OpenPCDet/output/kitti_models/pointpillar/default/eval/eval_with_train
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = max(args.epochs - 10, 0)  # Only evaluate the last 10 epochs
    # 3.调用test中的repeat_eval_ckpt进行模型评估
    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loader, args, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train
    )
    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
