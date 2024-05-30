import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

class RandomWalk:
    
    def __init__(self, lattice_points, num_steps):
        
        self.lattice_points = lattice_points
        self.num_steps = num_steps
    
    def ngon(self, center, radius):
        '''Creates an n sided polygon whose coreners serve as lattice points, i.e, the allowable points where the drunkard/particle can go.'''
        angles = np.linspace(0, 2*np.pi, self.lattice_points+1)[:-1]
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        shape = np.column_stack((x,y))
    
        return shape    
    
    def fixedStep(self, starting_position, step_size):
        '''This is the random walk with fixed step size.'''
        
        position = np.zeros((self.num_steps+1,2))
        position[0] = starting_position
        displacement = np.zeros((self.num_steps,))
        
        for i in range(self.num_steps):
            
            directions = self.ngon(position[-1], step_size)
            j = np.random.randint(0, directions.shape[0])
            position[i+1] = position[i] + directions[j]
            displacement[i] = np.linalg.norm(position[i] - starting_position)
        
        return position, displacement
    
    def trueRandomWalk(self, starting_position, step_min, step_max):
        '''This is random walk but the step size can vary between step_min and step_max at every step. This is closer to real life cases.'''
    
        position = np.zeros((self.num_steps+1,2))
        position[0] = starting_position
        displacement = np.zeros((self.num_steps,))
        
        for i in range(self.num_steps):
            
            rand_stepsize = np.random.rand(1) * (step_max - step_min)
            directions = self.ngon(position[-1], rand_stepsize)
            j = np.random.randint(0, directions.shape[0])
            position[i+1] = position[i] + directions[j]
            displacement[i] = np.linalg.norm(position[i] - starting_position)
        
        return position, displacement
    
    def plot_results(self, position, displacement):
        '''Function to plot the pathway and displacement as a function of time.'''
        
        plt.figure()
        plt.plot(position[:,0], position[:,1], '-o', color="blue", label = 'Pathway')
        plt.plot(position[0,0],position[0,1], '-o', color = 'green', label = 'Starting point')
        plt.plot(position[-1,0], position[-1,1], '-o', color="red", label = 'End point')
        
        for i in range(1, len(position)):
            plt.arrow(position[i-1, 0], position[i-1, 1], 
            position[i, 0] - position[i-1, 0], 
            position[i, 1] - position[i-1, 1], 
            head_width=0.1, length_includes_head=True, color='blue')
        
        plt.axis('equal')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid(True)
        plt.title('Path of Random Walk')
        plt.legend()
        
        x = range(self.num_steps)
        plt.figure()
        plt.plot(x, displacement, '-o' ,color="purple")
        plt.axhline(y=np.mean(displacement), color='red', linestyle='--', label=f'Average distance = {np.mean(displacement):.4f}')
        plt.annotate(f'Final distance = {displacement[-1]:.2f}', (x[-1], displacement[-1]),ha='center')
        plt.xlabel('Step')
        plt.ylabel('Distance')
        plt.title('Distance History')
        plt.grid(True)
        plt.legend()
        plt.draw()
        plt.show()

step_size = 1
number_of_steps = 100
drunkard = RandomWalk(1000,number_of_steps)

pos, disp = drunkard.fixedStep([0,0],step_size)
drunkard.plot_results(pos,disp)

# max_stepsize = 1
# min_stepsize = 0
# posTrue, dispTrue = drunkard.trueRandomWalk([0,0],min_stepsize, max_stepsize)
# drunkard.plot_results(posTrue,dispTrue)

theo_disp = step_size * np.sqrt(number_of_steps)
error = abs((disp[-1] - theo_disp)/theo_disp) * 100
print(f"The deviation of final distance from theoretical average value = {error:.3f}%")