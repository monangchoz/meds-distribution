import math

import numpy as np
from ep_heuristic.random_slpack import random_slpack
from line_profiler import profile
from problem.hvrp3l import HVRP3L
from problem.solution import NO_VEHICLE, Solution
from pymoo.core.problem import ElementwiseProblem
from pymoo_interface.arr1 import ARR1, RepairMechanism


class HVRP3L_OPT(ElementwiseProblem):
    def __init__(self,
                 hvrp3l_instance: HVRP3L,
                 repair_mechanism: RepairMechanism,
                 **kwargs):
        super().__init__(elementwise=True, **kwargs)
        self.hvrp3l_instance = hvrp3l_instance
        self.n_var = 2*hvrp3l_instance.num_customers
        self.repair = repair_mechanism.repair

        # first num_customers dims are for customer priority
        # the next num_customers dims are for vehicle assignment
        self.xl = np.zeros([self.n_var, ], dtype=float)
        self.xu = np.ones([self.n_var, ], dtype=float)
        
    # this is the decoding method
    @profile
    def _evaluate(self, x, out, *args, **kwargs):
        solution: Solution = Solution(self.hvrp3l_instance)
        problem = self.hvrp3l_instance
        customers = solution.problem.customers
        # try to map first into vehicle
        # if not feasible?
        
        for i in range(self.hvrp3l_instance.num_customers, 2*self.hvrp3l_instance.num_customers):
            ci = i-self.hvrp3l_instance.num_customers
            if customers[ci].need_refer_truck:
                vi = math.floor(x[i]*self.hvrp3l_instance.num_reefer_trucks)
            else:
                vi = math.floor(x[i]*self.hvrp3l_instance.num_vehicles)
            solution.cust_vhc_assignment_map[ci] = vi

        for vi in range(self.hvrp3l_instance.num_vehicles):
            vi_cust_idx = np.nonzero(solution.cust_vhc_assignment_map==vi)[0]
            if len(vi_cust_idx) == 0:
                continue

            # simple capacity checking
            total_weights = np.sum(solution.customer_demand_weights[vi_cust_idx])
            if total_weights>problem.vehicle_weight_capacities[vi]:
                continue
            total_volumes = np.sum(solution.customer_demand_volumes[vi_cust_idx])
            if total_volumes>problem.vehicle_volume_capacities[vi]:
                continue

            
            cust_priorities = x[vi_cust_idx]
            sorted_idx = np.argsort(cust_priorities)
            vi_cust_idx = vi_cust_idx[sorted_idx]

            total_num_items = sum([customers[ci].num_items for ci in vi_cust_idx])

            # this all actually can be pre-allocated in the problem interface
            # and used freely, to remove allocation time
            item_dims: np.ndarray = np.zeros([total_num_items, 3], dtype=float)
            item_volumes: np.ndarray = np.zeros([total_num_items, ], dtype=float)
            item_weights: np.ndarray = np.zeros([total_num_items, ], dtype=float)
            item_priorities: np.ndarray = np.zeros([total_num_items, ], dtype=float)
            n = 0
            for i, ci in enumerate(vi_cust_idx):
                c_num_items = customers[ci].num_items
                item_mask = problem.customer_item_mask[ci, :]
                item_dims[n:n+c_num_items] = problem.item_dims[item_mask]
                item_volumes[n:n+c_num_items] = problem.item_volumes[item_mask]
                item_weights[n:n+c_num_items] = problem.item_weights[item_mask]
                item_priorities[n:n+c_num_items] = i
                n += c_num_items
            
            # let's try packing
            container_dim = problem.vehicle_container_dims[vi]
            packing_result = random_slpack(item_dims,
                                           item_volumes,
                                           item_priorities,
                                           container_dim,
                                           0.8,
                                           5)
            positions, rotations, is_packing_feasible = packing_result
            
            if not is_packing_feasible:
                solution.cust_vhc_assignment_map[vi_cust_idx] = NO_VEHICLE
                continue
            # now commit the route, because we can pack
            solution.cust_vhc_assignment_map[vi_cust_idx] = vi
            solution.filled_volumes[vi] = total_volumes
            solution.filled_weight_caps[vi] = total_weights
            solution.routes[vi] += vi_cust_idx.tolist()
            n = 0
            for i, ci in enumerate(vi_cust_idx):
                c_num_items = customers[ci].num_items
                item_mask = problem.customer_item_mask[ci, :]
                solution.item_positions[item_mask] = positions[n:n+c_num_items]
                solution.item_rotations[item_mask] = rotations[n:n+c_num_items]
                n += c_num_items

        for vi in range(solution.num_vehicles):
            if len(solution.routes[vi]) == 1:
                continue
            solution.total_vehicle_fixed_cost += solution.vehicle_fixed_costs[vi]
            total_distance = solution.problem.compute_route_total_distance(solution.routes[vi])
            solution.total_vehicle_variable_cost += total_distance*solution.vehicle_variable_costs[vi]
        self.repair(solution)
        out["F"] = solution.total_cost
        # print(solution.total_cost)
        # try inserting, if not feasible, cancel this vehicle assignment, and 
        # then we repair later.
            