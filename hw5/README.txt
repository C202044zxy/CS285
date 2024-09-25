1) See hw1 if you'd like to see installation instructions. You do NOT have to redo them.
2) See the PDF for the rest of the instructions.

RND

python cs285/scripts/run_hw5_explore.py -cfg experiments/exploration/pointmass_easy_rnd.yaml --dataset_dir datasets

python cs285/scripts/run_hw5_explore.py -cfg experiments/exploration/pointmass_medium_rnd.yaml --dataset_dir datasets

python cs285/scripts/run_hw5_explore.py -cfg experiments/exploration/pointmass_hard_rnd.yaml --dataset_dir datasets

感觉最他宝贝的就是 pointmass 里面 epsilon 设置成了 2，然后 epsilon 是用来搞 done 的判断的，如果和终点的距离小于 epsilon 就会 terminate，我一直以为是我的 agent 过不了那个分界线，真的太宝贝离谱了，反正我是把 pointmass 看了一遍才知道问题在哪里，最后解决方案就是把 epsilon 调成 0.1 就行了（后面可以调回来）

CQL

注意把 RND 的数据拿过来的以后一定要 enable reward，要不然 reward 都没有学习个宝贝，改一下 run hw5 exploration 里面的代码就行了。

python ./cs285/scripts/run_hw5_offline.py -cfg experiments/offline/pointmass_easy_cql.yaml --dataset_dir datasets

python ./cs285/scripts/run_hw5_offline.py -cfg experiments/offline/pointmass_medium_cql.yaml --dataset_dir datasets

AWAC

就是 policy gradient，只不过 actor 并不是单纯基于 observation 训练的，而是基于 (observation, action) 对训练的。

改了好久发现一个问题，就是 get action 的时候需要用 actor 来，而不是取 critic 的 argmax，这是因为我们默认 actor 是 in-distribution 的，而 critic 里面还是可能涉及 OOD

python ./cs285/scripts/run_hw5_offline.py -cfg experiments/offline/pointmass_hard_awac.yaml --dataset_dir datasets

IQL

python ./cs285/scripts/run_hw5_offline.py -cfg experiments/offline/pointmass_medium_iql.yaml --dataset_dir datasets

这东西对数据比较敏感，还是需要开 20000 的数据集才可以训练，在 handout 的 4.3 中也有提示。所以 10000 的数据集跑不动可能真是数据集大小的问题。

Online Fine-tuning

python ./cs285/scripts/run_hw5_finetune.py -cfg experiments/finetuning/pointmass_hard_cql_finetune.yaml --dataset_dir datasets