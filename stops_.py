__author__ = 'Kamil Koziara'

import numpy
import pyopencl as cl


def mmap(fun, mat):
    f = numpy.vectorize(fun)
    return f(mat)


class Stops2:
    def __init__(self, gene_mat, pop, mul_mat, env_map, bound=None, secretion=None, reception=None, receptors=None,
                 init_env = None, secr_amount=1.0, leak=0.1, max_con = 1000.0, diffusion_rate = 0.0,
                 asym=None, asym_id=-3, div_id=-2, die_id=-1, opencl = False):
        """
        Init of Stops
        Parameters:
         - gene_mat - matrix of gene interactions [GENE_NUM, GENE_NUM]
         - pop - array with initial population [POP_SIZE, GENE_NUM]
         - mul_mat - matrix with adjacency graph between each environment [ENV_SIZE, ENV_SIZE], number on [i,j]
                    is factor of interaction between environment i and j, value is float in (0,1)
         - env_map - array mapping cells to environments
         - bound - vector of max value of each gene [GENE_NUM]
         - secretion - vector of length LIG_NUM where secretion[i] contains index
            of a gene which must be on to secrete ligand i
         - reception - vector of length LIG_NUM where reception[i] contains index
            of a gene which will be set to on when ligand i is accepted
         - receptors - vector of length LIG_NUM where receptors[i] contains index
            of a gene which has to be on to accept ligand i; special value -1 means that there is no
            need for specific gene expression for the ligand
         - secr_amount - amount of ligand secreted to the environment each time
         - leak - amount of ligand leaking from the environment each time
         - max_con - maximal ligand concentration
         - asym - array [2, GENE_NUM] specifying how to modify the both children states, children states are created
            by multiplying parent state by these vectors
         - asym_id - id of gene responsible for asymmetric division
         - div_id - id of gene responsible for division
         - die_id - id of gene responsible for death
         - opencl - if set to True opencl is used to boost the speed
        """
        self.gene_mat = numpy.array(gene_mat).astype(numpy.float32)
        self.pop = numpy.array(pop).astype(numpy.float32)
        self.mul_mat = numpy.array(mul_mat).astype(numpy.float32)
        self.secr_amount = secr_amount
        self.leak = leak
        self.max_con = max_con
        self.row_size = self.gene_mat.shape[0]
        self.pop_size = self.pop.shape[0]
        self.env_size = self.mul_mat.shape[0]

        self.diffusion_rate = diffusion_rate

        if bound != None:
            self.bound = numpy.array(bound).astype(numpy.float32)
        else:
            # bound default - all ones
            self.bound = numpy.ones(self.row_size).astype(numpy.float32)

        if secretion != None:
            self.secretion = numpy.array(secretion).astype(numpy.int32)
        else:
            self.secretion = numpy.array([]).astype(numpy.int32)

        if reception != None:
            self.reception = numpy.array(reception).astype(numpy.int32)
        else:
            self.reception = numpy.array([]).astype(numpy.int32)

        self.max_lig = len(secretion)

        if init_env is None:
            self.init_env = numpy.zeros(self.max_lig)
        else:
            self.init_env = init_env

        self.env = numpy.array([self.init_env] * self.env_size).astype(numpy.float32)
        self.env_map = numpy.array(env_map).astype(numpy.int32)
        self.env_count = numpy.zeros(self.env_size).astype(numpy.int32)

        for i in self.env_map:
            self.env_count[i] += 1

        if receptors != None:
            self.receptors = numpy.array(receptors).astype(numpy.int32)
        else:
            # receptors - default value "-1" - no receptor for ligand is necessary
            self.receptors = numpy.array([-1] * self.max_lig).astype(numpy.int32)

        self._random = numpy.random.random

        if asym != None:
            self.asym = numpy.array(asym).astype(numpy.int32)
        else:
            self.asym = numpy.ones([2, self.row_size]).astype(numpy.int32)

        self.asym_id = asym_id
        self.div_id = div_id
        self.die_id = die_id

        self.nbors = []
        for i, row in enumerate(self.mul_mat):
            i_nbors = []
            for j, x in enumerate(row):
                if x > 0:
                    i_nbors.append((j, x))
            self.nbors.append(i_nbors)


        self.opencl = opencl

        if opencl:
            self.ctx = cl.create_some_context()
            self.queue = cl.CommandQueue(self.ctx)
            self.mf = cl.mem_flags
            #init kernel
            self.program = self.__prepare_kernel()
            self.rand_state_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE, size = self.pop_size * 112)
            self.program.init_ranlux(self.queue, (self.pop_size, 1), None,
                                     numpy.uint32(numpy.random.randint(4e10)), self.rand_state_buf)
            # prepare multiplication matrix
            self.mul_mat_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.mul_mat)


    def step(self):
        if self.opencl:
            self._step_opencl()
        else:
            self._step_numpy()

    def __prepare_kernel(self):
        with open("mat_mul_kernel.c") as f:
            mat_mul_kernel = f.read()
        with open("ranlux_random.c") as f:
            rand_fun = f.read()
        with open("expression_kernel.c") as f:
            expr_kernel = f.read()
        with open("secretion_kernel.c") as f:
            secr_kernel = f.read()
        with open("reception_kernel.c") as f:
            rec_kernel = f.read()

        params = """#define MAX_LIG %(max_lig)d
                    #define ROW_SIZE %(row_size)d
                    #define ENV_SIZE %(env_size)d
                 """ % {"row_size": self.row_size, "env_size": self.env_size,
                                                         "max_lig": self.max_lig}

        # dbg = "# pragma OPENCL EXTENSION cl_intel_printf :enable\n"
        return cl.Program(self.ctx, params +
            mat_mul_kernel + "\n" +
            rand_fun + "\n" +
            expr_kernel + "\n" +
            secr_kernel + "\n" +
            rec_kernel).build()

    def _step_opencl(self): #TODO: move all used-in-all-steps inits up
        # expression
        pop_size = self.pop_size
        gene_mat_size = self.gene_mat.shape[0]

        pop_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.pop)
        gene_mat_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.gene_mat)
        tokens_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE, size = 4 * pop_size * gene_mat_size)
        # generate matrix of tokens simulating probability of particular actions taken by a cell
        # generate one random number for each cell
        self.program.mat_mul(self.queue, (pop_size, gene_mat_size), None, tokens_buf,
                     pop_buf, gene_mat_buf, numpy.int32(gene_mat_size))
        rand_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE, size = 4 * pop_size)
        self.program.get_random_vector(self.queue, (int(pop_size / 4 + 1), 1), None,
                                  rand_buf, numpy.int32(pop_size), self.rand_state_buf)
        bound_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.bound)
        # generating new population state
        self.program.choice(self.queue, (pop_size, 1), None, pop_buf, tokens_buf, rand_buf, bound_buf)

        # secretion
        env_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.env)
        env_map_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.env_map)
        secr_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.secretion)

        self.program.secretion(self.queue, (self.pop_size, 1), None, pop_buf, env_buf, env_map_buf, secr_buf,
                               numpy.float32(self.max_con), numpy.float32(self.secr_amount))
        self.program.leaking(self.queue, (self.env_size, 1), None, env_buf, numpy.float32(self.leak))

        # reception
        env_muls_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE, size = self.env_size * 4)
        env_count_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = self.env_count)
        self.program.calculate_env_muls(self.queue, (self.env_size, 1), None,
                                        env_muls_buf,self.mul_mat_buf, env_count_buf)

        pop_hit_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE, size = self.pop_size * self.max_lig * 4)
        self.program.fill_buffer(self.queue, (self.pop_size * self.max_lig, 1), None, pop_hit_buf, numpy.int32(0))

        rec_gene_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.reception)
        receptors_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.receptors)

        self.program.reception(self.queue, (self.env_size, 1), None,
                               pop_hit_buf, env_buf, self.mul_mat_buf, env_muls_buf,
                               env_map_buf, pop_buf,
                               receptors_buf, numpy.int32(self.pop_size), self.rand_state_buf)

        self.program.update_pop_with_reception(self.queue, (pop_size, 1), None,
                                               pop_buf, pop_hit_buf, rec_gene_buf, bound_buf)

        # storing state
        cl.enqueue_copy(self.queue, self.env, env_buf)
        cl.enqueue_copy(self.queue, self.pop, pop_buf)

        # division/death done by host (no heavy computations there)
        self._divdie()

        if self.diffusion_rate > 0:
            self._diffusion() #done by the host (to be moved to the gpu)


    def _expression(self):
        # generate matrix of tokens simulating probability of particular actions taken by a cell
        # generate one random number for each cell
        tokens_mat = self.pop.dot(self.gene_mat)
        rnd_mat = self._random(self.pop_size) # random number for each cell

        # cumulative influence by cell
        sel_mat = numpy.cumsum(abs(tokens_mat), axis=1)
        # total influence by cell
        sums = numpy.sum(abs(tokens_mat), axis=1).reshape(self.pop_size, 1)
        # normalized influence by cell
        norm_mat = numpy.array(sel_mat, dtype=numpy.float32) / sums
        # as a vertical vector
        rnd_mat.resize((self.pop_size, 1))
        # boolean matrix with values greater than random
        bool_mat = (norm_mat - rnd_mat) > 0
        ind_mat = numpy.resize(numpy.array(range(self.pop.shape[1]) * self.pop_size) + 1, self.pop.shape)
        # matrix of indices
        sel_arr = numpy.select(list(bool_mat.transpose()), list(ind_mat.transpose())) - 1
        # the index of the first value greater than random (-1 if no such value)
        dir_arr = numpy.select(list(bool_mat.transpose()), list(numpy.array(tokens_mat).transpose()))
        for i, (s, d) in enumerate(zip(sel_arr, dir_arr)):
            if s >= 0:
                self.pop[i, s] = max(0, min(self.bound[s], self.pop[i, s] + (d / abs(d))))

    def _secretion(self):
        # secretion
        for i in range(self.pop_size):
            # for each cell
            for j, k in enumerate(self.secretion):
                # for each ligand
                if self.pop[i, k] > 0:
                    # if ligand is expressed
                    ei = self.env_map[i]
                    self.env[ei, j] = min(self.max_con, self.env[ei, j] + self.secr_amount)
                    self.pop[i, k] -= 1.0 # or get down to 0?

        # leaking
        leak_fun = numpy.vectorize(lambda x : max(0.0, x - self.leak))
        self.env = leak_fun(self.env)

    def _diffusion(self):
        for j, lig in enumerate(self.env.T):
            sum = 0.0
            df_sum = 0.0
            new_env = numpy.zeros(self.env.shape[0]).astype(numpy.float32)
            for i, val in enumerate(lig):
                # if i != j TODO
                sum += val
                wsum = 1.0
                nval = val
                for k, w in self.nbors[i]:
                    w *= self.diffusion_rate
                    wsum += w
                    nval += lig[k] * w
                nval /= wsum
                df_sum += nval
                new_env[i] = nval
            if df_sum > 0:
                self.env[:,j] = new_env * sum / df_sum



    def _reception(self):
        # reception
        env_muls = numpy.zeros(self.env_size)
        for i, mul_row in enumerate(self.mul_mat):
            #for each env
            con_sum = 1.0 / numpy.sum(self.env_count * mul_row)
            if not (mul_row == 1).any():
                con_sum *= numpy.product(numpy.power(mul_row, self.env_count))
            env_muls[i] = con_sum


        for j, k in enumerate(self.reception):
            # for each ligand j that can be absorbed
            env_ligand = self.env[:, j] * env_muls
            env_mod = numpy.zeros(self.env_size)

            for i, pop_row in enumerate(self.pop):
                # and for each cell  calculate probs of receiving ligand from envs
                if self.can_receive(j, pop_row):
                    ie = self.env_map[i]
                    rec_probs = env_ligand * self.mul_mat[ie]
                    is_received = rec_probs > self._random(self.env_size).astype(numpy.float32)
                    if is_received.any():
                        self.pop[i, k] = min(self.pop[i, k] + 1, self.bound[k])
                        env_mod += is_received

            # removing absorbed ligands
            for i, new_lig in enumerate(self.env[:, j] - env_mod):
                self.env[i, j] = max(0, new_lig)

    def _divdie(self):
        g_act = 1.0 # TODO
        # count net change
        div = numpy.sum(self.pop[:, self.div_id] >= g_act)
        die = numpy.sum(self.pop[:, self.die_id] >= g_act)
        if div != 0 or die != 0:
            print div - die
            change = div - die
            self.pop_size += change
            new_pop = numpy.zeros([self.pop_size, self.row_size])
            new_env_map = numpy.zeros(self.pop_size)

            j = 0 # index in self.pop2
            for i, row in enumerate(self.pop):
                if row[self.die_id] >= g_act: # dying
                    pass #forget this guy
                elif row[self.div_id] >= g_act: # dividing
                    row[self.div_id] = 0.0
                    if row[self.asym_id] >= g_act: # assymetric
                        new_pop[j] = numpy.minimum(self.bound, row * self.asym[0])
                        new_pop[j + 1] = numpy.minimum(self.bound, row * self.asym[1])
                    else: # symmetric
                        new_pop[j] = row
                        new_pop[j + 1] = row

                    new_env_map[j] = self.env_map[i]
                    new_env_map[j + 1] = self.env_map[i]
                    j += 2
                else:
                    # without division or death
                    new_pop[j] = row
                    new_env_map[j] = self.env_map[i]
                    j += 1

            self.pop = new_pop.astype(numpy.float32)
            self.env_map = new_env_map.astype(numpy.int32)
            self.env_count = numpy.zeros(self.env_size).astype(numpy.int32)
            for i in self.env_map:
                self.env_count[i] += 1

    def _step_numpy(self):
        self._expression()
        self._secretion()
        self._reception()
        self._divdie()
        if self.diffusion_rate > 0:
            self._diffusion()

    def sim(self, steps=100):
        for i in range(steps):
            self.step()

    def can_receive(self, ligand, row):
        """Function describes if a specific cell (defined by its state) can receive specified ligand"""
        rec = self.receptors[ligand]
        return rec == -1 or row[rec] > 0
