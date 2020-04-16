import matplotlib.pyplot as plt


class CustomCyclicLR:

    def __init__(self, base_lr, max_lr, step_size, iterations):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.iterations = iterations
        self.lr = []
        self.pad_factor = max_lr / 10

    def cycle(self, iteration):
        return int(1 + (iteration / (2 * self.step_size)))

    def lr_position(self, iteration, cycle):
        return abs(iteration / self.step_size - 2 * cycle + 1)

    def current_lr(self, lr_position):
        return self.base_lr + (self.max_lr - self.base_lr) * (1 - lr_position)

    def cyclic_lr(self, plot=True):
        for iteration in range(self.iterations):
            cycle = self.cycle(iteration)
            lr_position = self.lr_position(iteration, cycle)
            self.lr.append(self.current_lr(lr_position))
        if plot:
            self.plot()
    
    def plot(self):
        # Initialize a figure
        fig = plt.figure(figsize=(10, 3))

        # Set plot title
        plt.title('Cyclic LR')

        # Label axes
        plt.xlabel('Iterations')
        plt.ylabel('Learning Rate')

        # Plot max lr line
        plt.axhline(self.max_lr, 0.03, 0.97, label='lr_max', color='r')
        plt.text(0, self.max_lr + self.pad_factor, 'lr_max')

        # Plot min lr line
        plt.axhline(self.base_lr, 0.03, 0.97, label='base_lr', color='r')
        plt.text(0, self.base_lr - self.pad_factor, 'base_lr')

        # Plot lr change
        plt.plot(self.lr)

        # Plot margins and save plot
        plt.margins(y=0.2)
        plt.tight_layout()
        plt.savefig('cyclic_lr_plot.png')