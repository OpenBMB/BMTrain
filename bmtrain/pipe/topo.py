class topology:
    def __init__(self,**config):
        # pipe_idx is the idx of the pipeline in the group
        self.rank = config['rank']
        pp_size = config["pipe_size"]
        tp_size = config["tp_size"]
        world_size = config["world_size"]
        assert world_size % (pp_size * tp_size) == 0, "The nums of GPUs must be divisible by the pipeline parallel size * tensor parallel size"

        dp_size = world_size // (pp_size * tp_size)
        config['tp_zero_size'] = dp_size
        config['zero_size'] = world_size // pp_size 
        self.pipe_size = config['pipe_size']
        
        stage_size = world_size // pp_size
        for i in range(world_size):
            self.pipe_idx = self.rank % stage_size 
            self.pipe_rank = self.rank // stage_size 
            self.tp_id = self.rank % tp_size
            self.tp_idx = self.rank // tp_size 
            #pp->zero
            self.pp_zero_idx = self.pipe_rank 
            self.pp_zero_id = self.pipe_idx 
            #tp->zero
            self.tp_zero_idx = self.tp_id 
            self.tp_zero_id = self.tp_idx
            #pp->tp->zero
            self.pp_tp_zero_idx = self.pipe_rank * tp_size + self.tp_id 
            self.pp_tp_zero_id = self.pipe_idx // tp_size
        #only zero
        self.zero_idx = 0
        self.zero_id = self.rank


    def get_group_id(self,group_name):
        if group_name == "pipe":
            return self.pipe_idx
        elif group_name == "zero":
            return self.zero_idx
        elif group_name == "tp_zero":
            return self.tp_zero_idx
        elif group_name == "tp":
            return self.tp_idx
        
    def get_group_rank(self,group_name):
        if group_name == "pipe":
            return self.pipe_rank
        elif group_name == "zero":
            return self.zero_id
        elif group_name == "tp_zero":
            return self.tp_zero_id
        elif group_name == "tp":
            return self.tp_id
        
    def get_peer(self, group_name, next_prev):
        if group_name == "pipe":
            if next_prev == "next":
                return self.pipe_rank+1 if self.pipe_rank < self.pipe_size - 1 else -1
            elif next_prev == "prev":
                return self.pipe_rank-1 if self.pipe_rank > 0 else -1
        elif group_name == "zero":
            if next_prev == "next":
                return self.zero_id+1 if self.zero_id < self.pipe_size - 1 else -1
            elif next_prev == "prev":
                return self.zero_id-1 if self.zero_id > 0 else -1
        elif group_name == "tp_zero":
            if next_prev == "next":
                return self.tp_zero_id+1 if self.tp_zero_id < self.pipe_size - 1 else -1
            elif next_prev == "prev":
                return self.tp_zero_id-1 if self.tp_zero_id > 0 else -1
        elif group_name == "tp":
            if next_prev == "next":
                return self.tp_id+1 if self.tp_id < self.pipe_size - 1 else -1
            elif next_prev == "prev":
                return self.tp_id-1 if self.tp_id > 0 else -1
        return -1


if __name__ == "__main__":
    topology1 = topology(**{"rank":0,"pipe_size":4,"tp_size":8,"world_size":32})
    topology2 = topology(**{"rank":8,"pipe_size":4,"tp_size":8,"world_size":32})
    topology3 = topology(**{"rank":16,"pipe_size":4,"tp_size":8,"world_size":32})
    topology4 = topology(**{"rank":24,"pipe_size":4,"tp_size":8,"world_size":32})
    from IPython import embed;embed()
    