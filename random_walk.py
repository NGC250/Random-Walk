import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

class RandomWalk:
    
    def __init__(self, allowed_directions, num_steps):
        '''allowed directions refers to the allowed points where the particle can travel (see ngon). 1000 is sufficient for any direction of movement.'''
        
        self.lattice_points = allowed_directions
        self.num_steps = num_steps
    
    def ngon(self, radius):
        '''Creates an n sided polygon whose corners serve as lattice points, i.e, the allowed directions'''
        
        angles = np.linspace(0, 2*np.pi, self.lattice_points+1)[:-1]
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        shape = np.column_stack((x,y))
    
        return shape    
    
    def fixedStep(self, starting_position, step_size):
        '''This is the random walk with fixed step size.'''
        
        direction_indices = np.random.randint(0, self.lattice_points, (self.num_steps,))
        directions = self.ngon(step_size)[direction_indices]
        position = np.zeros((self.num_steps+1,2))
        displacement = np.zeros((self.num_steps,))
        
        position[0] = starting_position
        for i in range(self.num_steps):
            position[i+1] = position[i] + directions[i]
            displacement[i] = np.linalg.norm(position[i] - position[0])
        
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

class Analysis:
    
    def __init__(self, starting_position, step_size):
        
        self.step_size = step_size
        self.starting_position = starting_position 
    
    def AverageDistance(self, max_steps, allowed_directions, iterations_to_convergence):
        '''this function calculates the average displacement of the particle for number_of_steps = 1 to max_steps'''
        
        final_distance_per_stepsize = np.zeros((max_steps,2))
        final_distance_per_cycle = np.zeros((iterations_to_convergence,))
        
        for i in range(max_steps):
            instance1 = RandomWalk(allowed_directions, i)
            for j in range(iterations_to_convergence):
                position, _ = instance1.fixedStep(self.starting_position, self.step_size)
                final_distance_per_cycle[j] = np.linalg.norm(position[-1] - position[0])
            final_distance_per_stepsize[i] = [i+1, np.mean(final_distance_per_cycle)]

        return final_distance_per_stepsize
    
    def CustomFunction(self, x, a, b):
        '''mathematical average displacement function to help curve fit'''
        return a * x**b
    
    def CurveFit(self, x_data, y_data, initial_guess):
        '''fits observed data to mathematical equation'''
        
        popt, pcov = sp.optimize.curve_fit(self.CustomFunction, x_data, y_data, p0 = initial_guess)
        
        y_fit = self.CustomFunction(x_data, *popt)
        rmse = np.sqrt(np.mean((y_data - y_fit) ** 2))
        ss_res = np.sum((y_data - y_fit) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print("Fit report:")
        print(f'RMSE = {rmse}')
        print(f'R^2 = {r_squared}')
        
        return popt, pcov
    
    def PlotFit(self, data, initial_guess):
        '''plots fitted and theoretical(mathematical) curves'''
        
        popt , _ = self.CurveFit(data[:,0], data[:,1], initial_guess)
        
        plt.figure()
        plt.scatter(data[:,0], data[:,1],s=1, color='indigo')
        plt.plot(data[:,0], self.CustomFunction(data[:,0], *popt), label='Fitted curve', color='purple')
        
        a_opt, b_opt = popt
        equation_text1 = f'Fitted curve(purple): {a_opt:.3f} * N^{b_opt:.3f}'
        equation_text2 = f'Theoretical curve(blue): {(np.sqrt(np.pi) / 2):.3f} * N^{0.5}'
        print(f"Best-fit parameters for a*N**b: [a b] = {popt}")
        
        plt.text(0.05, 0.95, equation_text1, transform=plt.gca().transAxes, fontsize=9, verticalalignment='top')
        plt.text(0.05, 0.90, equation_text2, transform=plt.gca().transAxes, fontsize=9, verticalalignment='top')
        
        plt.plot(data[:,0], self.step_size * sp.special.gamma(3/2) * np.sqrt(data[:,0]), label='Theoretical curve')
        
        plt.legend(loc='lower right')
        plt.draw()
        plt.show()   

def main():
    
    starting_position = [0,0]
    step_size = 1
    number_of_steps = 100
    max_steps = 75 #maximum number of steps for curve fitting analysis. Not the same as number_of_steps
    allowed_points = 1000  #1000 = cirle , i.e., particle is free to move in any direction in 2D plane.
    iterations_to_convergence = 75
    
    # drunkard = RandomWalk(allowed_points , number_of_steps)
    # pos, disp = drunkard.fixedStep(starting_position , step_size)
    # drunkard.plot_results(pos,disp)
    
    A1 = Analysis(starting_position, step_size)
    dist_data = A1.AverageDistance(max_steps, allowed_points, iterations_to_convergence)
    A1.PlotFit(dist_data, [1,1])

if __name__ == '__main__':
    main()
