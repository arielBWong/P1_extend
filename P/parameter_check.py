import json
import numpy as np
from surrogate_problems import branin, GPc, Gomez3, Mystery, Reverse_Mystery, SHCBc, HS100, Haupt_schewefel, \
    MO_linearTest, single_krg_optim, WFG, iDTLZ, DTLZs, ego_fitness, EI, MAF

if __name__ == "__main__":


    for i in np.arange(0, 1):
        for j in np.arange(1, 3):
            problem_setting = 'half' + str(j) + '_problems_nd_' + str(i) + '.json'
            with open(problem_setting, 'r') as data_file:
                hyp = json.load(data_file)

                target_problems = hyp['MO_target_problems']
                search_ideal = hyp['search_ideal']

                max_eval = hyp['max_eval']
                num_pop = hyp['num_pop']
                num_gen = hyp['num_gen']

                if num_gen != 100:
                    print(problem_setting)
                    print('num_gen is wrong!')


                if num_pop != 100:
                    print(problem_setting)
                    print('num_pop is wrong')

                if max_eval != 300:
                    print(problem_setting)
                    print('max_eval is wrong')

                if len(target_problems) != 10:
                    print(problem_setting)
                    print('target_problems length is wrong')


                for target_problem in target_problems:
                    prob = eval(target_problem)
                    if prob.n_obj != 3:
                        print(problem_setting)
                        print('target_problems objective nuber  is wrong')

                if j == 1:
                  probstr = ['MAF1', 'MAF2', 'MAF3', 'MAF4',  'MAF5', 'MAF6', 'DTLZ1',   'DTLZ2', 'DTLZ3', 'DTLZ4']
                  for m, k in enumerate(target_problems):
                      if probstr[m] not in k:
                          print(problem_setting)
                          print(k)
                if j == 2:
                    probstr = ['DTLZ7', 'WFG_1','WFG_2', 'WFG_3', 'WFG_4', 'WFG_5', 'WFG_6', 'WFG_7','WFG_8', 'WFG_9']
                    for m, k in enumerate(target_problems):
                        if probstr[m] not in k:
                            print(problem_setting)
                            print(k)





