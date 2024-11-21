class JitterNoiseScheduler:
    def __init__(self, base_noise, target_noise, start_step, end_step):
        self.base_noise = base_noise
        self.target_noise = target_noise
        self.start_step = start_step
        self.end_step = end_step
        self.current_step = 0

    def update(self):
        """Calculate scheduled noise value based on current training step."""
        self.current_step += 1
        
        if self.current_step < self.start_step or self.end_step <= self.start_step:
            return self.base_noise
        
        if self.current_step >= self.end_step:
            return self.target_noise
        
        # Linear interpolation between base_noise and target_noise
        progress = (self.current_step - self.start_step) / (self.end_step - self.start_step)
        return self.base_noise + (self.target_noise - self.base_noise) * progress