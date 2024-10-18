from hijacobi.model import *
from hijacobi.solver import *


class Task():

    def __init__(self, BH: Spacetime, eq_data: list):
        '''
        eq_data: list of dicts {'r_s': val, 'r_obs': val,  'th_s': val, 'th_obs': val, 'n_th': val, 'n_r': val}
        '''

        self.BH = BH

        self.eq_data = eq_data
        self.N_eq = len(eq_data)

        self.sol = None


    def set_solver(self, solver: SolveIEQ):
        
        self.sol = solver


    def get_solution_frame(self):

        outp = self.eq_data.copy()

        for i in range(self.N_eq):
            outp[i]['res'] = None

        return outp


    def make_eq(self, eqN: int):

        th_order = self.eq_data[eqN]['n_th']
        r_order = self.eq_data[eqN]['n_r']

        th_s = self.eq_data[eqN]['th_s']
        th_obs = self.eq_data[eqN]['th_obs']

        r_s = self.eq_data[eqN]['r_s']
        r_obs = self.eq_data[eqN]['r_obs']

        eq = None


        if (th_order == 0) & (r_order == 0):
            eq = lambda l, q: self.sol.Ir([r_s, r_obs], l, q) - (self.sol.Ith([th_obs, th_s], l, q))


        if (th_order == 1) & (r_order == 0):
            eq = lambda l, q: self.sol.Ir([r_s, r_obs], l, q) - \
                 torch.abs(self.BH.Ith_t(th_s, l, q) + self.BH.Ith_t(th_obs, l, q))


        return eq
    

    def log(self):
        #Написать, какие уравнения для каких параметров решаются"
        # r_obs = 300, th_obs = 0.5 pi
        # Ith(th_obs, th_i) + Ir(r_obs, r_i) = 0 : (r1=2.1, th1=0.5pi), (r2=2.2, th2=0.6pi)
        # или
        # Ith(th_obs, th_i) + Ir(r_obs, r_i) = 0 : (r1=[2.1, 2.3], th1=0.5pi)
        # Ith(th_obs, th_turn) - Ith(th_turn, th_i) + ... : TP_theta, (...)
        pass


    def solve(self):

        res = self.get_solution_frame()

        #eq0 = self.make_eq(0, 0)
        #self.sol.prepare(eq0)

        for n in range(self.N_eq):

            task_inps = self.eq_data[n]

            print('Solving Eq #', n, ' conditions: ',  task_inps)
            eq = self.make_eq(n)
            
            parA, parB = self.sol.solve(eq, task_inps)

            res[n]['res'] = [parA, parB]
            #res[n]['err'] = err

        return res


    