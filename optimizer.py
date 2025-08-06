# optimizer.py
import torch

class SimulationOptimizer:
    def __init__(self, initial_params, target_data, sim_function):
        '''
        initial_params: dict of parameter names to initial values (e.g., {"E": 5000, "friction_coeff": 0.3, ...})
        target_data: dict of target outputs to match (e.g., {"force_profile": [...], "depth_profile": [...]})
        sim_function: function that runs the simulation given a parameter dict and returns a dict of results
        '''
        self.target_data = target_data
        # Flatten parameters into a tensor for optimization
        self.param_names = list(initial_params.keys())
        self.params = torch.tensor([initial_params[name] for name in self.param_names], 
                                   requires_grad=True, dtype=torch.float32)
        self.sim_function = sim_function
        # Optimizer (Adam) setup
        self.optimizer = torch.optim.Adam([self.params], lr=0.1)
    
    def objective(self, param_values):
        # Run simulation with given parameter values and compute a loss against target_data
        # Convert param_values (numpy array) into a parameter dict
        param_dict = {name: float(param_values[i]) for i, name in enumerate(self.param_names)}
        sim_results = self.sim_function(param_dict)  # user-provided simulation run with these params
        loss = 0.0
        # Check for constraint violations in sim_results
        if sim_results.get('unstable', False):
            loss += 1e3  # heavy penalty for instability (particle explosion, etc.)
        if sim_results.get('vol_change', 0.0) > 0.1:
            loss += 100.0  # penalty if volume change >10%
        if sim_results.get('max_penetration', 0.0) > sim_results.get('penetration_thresh', float('inf')):
            loss += 100.0  # penalty if penetration beyond allowed threshold
        # Compare simulation force profile to target force profile (if available)
        if 'force_profile' in sim_results and 'force_profile' in self.target_data:
            # e.g., mean squared error over the force vs time curve
            diff = torch.tensor(sim_results['force_profile']) - torch.tensor(self.target_data['force_profile'])
            loss += float((diff ** 2).mean())
        # Compare deformation (penetration depth) profile if provided
        if 'depth_profile' in sim_results and 'depth_profile' in self.target_data:
            diff = torch.tensor(sim_results['depth_profile']) - torch.tensor(self.target_data['depth_profile'])
            loss += float((diff ** 2).mean())
        return loss
    
    def run(self, iterations=20):
        best_loss = float('inf')
        best_params = None
        for it in range(iterations):
            # Compute objective (loss) for current parameters (no autograd, we'll do finite diff)
            current_values = self.params.detach().numpy()
            base_loss = self.objective(current_values)
            # Finite difference gradient approximation
            grad = torch.zeros_like(self.params)
            eps = 1e-3
            for i in range(len(self.params)):
                original = current_values[i]
                # loss at param + eps
                current_values[i] = original + eps
                loss_plus = self.objective(current_values)
                # loss at param - eps
                current_values[i] = original - eps
                loss_minus = self.objective(current_values)
                # restore original
                current_values[i] = original
                grad[i] = (loss_plus - loss_minus) / (2 * eps)
            # Apply Adam optimizer step with this gradient
            self.optimizer.zero_grad()
            self.params.grad = grad  # assign computed gradient
            self.optimizer.step()
            # Clamp parameters to valid ranges (e.g., non-negative values for physical params)
            with torch.no_grad():
                self.params[:] = torch.clamp(self.params, min=0.0)
            # Track the best parameters
            if base_loss < best_loss:
                best_loss = base_loss
                best_params = self.params.detach().numpy().copy()
            print(f"Iteration {it}: loss = {base_loss:.4f}, params = {self.params.detach().numpy()}")
        # Return best found parameters as a dictionary
        optimized = {name: best_params[i] for i, name in enumerate(self.param_names)}
        return optimized
