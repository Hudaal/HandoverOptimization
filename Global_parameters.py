
class Global_parameters:
    def __init__(self) -> None:
        self.num_iterations = 100000 

        self.initial_collect_steps = 500 
        self.collect_steps_per_iteration = 1 
        self.replay_buffer_capacity = 10000 

        self.batch_size = 5 

        self.critic_learning_rate = 1e-3 
        self.actor_learning_rate = 1e-4 
        self.alpha_learning_rate = 1e-3 
        self.target_update_tau = 0.005 
        self.target_update_period = 1 
        self.gamma = 0.99999999 
        self.reward_scale_factor = 1.0 

        self.others = 4
        self.all_count = 11
        self.UE_upper_count = 18

        self.log_interval = 1 

        self.eval_interval = 10 
        self.eval2_interval = 15

        self.policy_save_interval = 500 

        self.ENB_Count = 5
        self.upper_limit = 34
        self.lower_limit = 0

        self.all_sate = (self.ENB_Count*self.all_count)+self.others

        self.actor_fc_layer_params = (self.all_sate, self.all_sate)
        self.critic_joint_fc_layer_params = (self.all_sate, self.all_sate)

        self.handovers_count = 0
        self.actions = []
        self.action__ = []
        self.reward_ = 0
        self.throughput_to_save = []
        self.rsrq_to_save = []
        self.cell_rsrqs = []
        self.cell_throughputs = []
        self.max_speed = 0
        self.min_speed = 0
        self.duration = 0
        self.ues = 0

        self.eval_actions = []
        self.eval_reward_ = 0
        self.eval_throughput_to_save = []
        self.eval_handovers_count = 0
        self.eval_action__ = []
        self.eval_rsrq_to_save = []
        self.eval_cell_rsrqs = []
        self.eval_cell_throughputs = []
        self.eval_max_speed = 0
        self.eval_min_speed = 0

        self.eval2_actions = []
        self.eval2_reward_ = 0
        self.eval2_throughput_to_save = []
        self.eval2_handovers_count = 0
        self.eval2_action__ = []
        self.eval2_rsrq_to_save = []
        self.eval2_cell_rsrqs = []
        self.eval2_cell_throughputs = []
        self.eval2_max_speed = 0
        self.eval2_min_speed = 0

        self.file_name = 'output/logFile.txt'
        self.events_file_name = 'output/simulatorFile.txt'
        self.rsrq_throughput_file = 'output/qualityValues.txt'

gp = Global_parameters()