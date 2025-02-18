
#import necessary modules/shortcuts
#array functions
import numpy as np
from numpy import pi
#math functions
import math
from math import factorial
#plotting functions
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
#scientific functions
import scipy
import scipy.special as special
#timer
from time import perf_counter

# Start the timer
t_start = perf_counter() 

#define the bohr radius
a_0 = 5.29e-11

#probability function for the 1s orbital
def orbital_probability(n,l,m):
    '''This function computes a generalized probability density for the 1s orbital'''
    #calculates the radial component of the wavefunction using scipy leguerre polynomials
    radial_comp = (2*r_array/n)**l * np.exp(-r_array/n) * scipy.special.genlaguerre(n-l-1,2*l+1)(2*r_array/n)
    #probability is the abs. square of a wavefunction
    radprob = abs(np.conj(radial_comp)*radial_comp)
    #normalize the probability
    radprob /= radprob.sum()
    
    #calculates the complete wavefunction using the radial component and scipy spherical harmonics
    wave_func = np.sqrt(((2/n*a_0)**3)*(factorial(n-l-1)/factorial(2*n*(n+l)))) * radial_comp * scipy.special.sph_harm(m,l,phi_array,theta_array) 
    #probability is the abs. of a wavefunctions complex conjugate times the wavefunction
    waveprob = abs(np.conj(wave_func)*wave_func)
    #normalize the probability
    waveprob /= waveprob.sum()
    #returns the full wavefunction probability distribution as well as its radial component
    return waveprob, radprob

#defining the x,y,z
x=np.linspace(-50,50,40)
y=np.linspace(-50,50,40)
z=np.linspace(-50,50,40)

#generate 3D arrays for the x,y,z axis
x_array, y_array, z_array =np.meshgrid(x,y,z)

#calculates the radius at each point in the grid
r_array = np.sqrt(np.square(x_array)+np.square(y_array)+np.square(z_array))
#calculates the theta value for each point in the grid
phi_array = np.arctan(y_array/x_array)
#calculates the phi value for each point in the grid
theta_array = np.arctan(np.sqrt(x_array**2+y_array**2)/x_array)

#creates 1D arrays using the elements of our grid arrays
D_x = x_array.flatten()
D_y = y_array.flatten()
D_z = z_array.flatten()

#stacks and then flattens our 1D arrays and converts to a list so that coordinates can be assembled into a new list
coord_list1D = np.ndarray.flatten(np.vstack([D_z,D_y,D_x]),'F').tolist()

#groups each every combination of x,y,z coords as a string to be passed to np.random.choice
coord_nestlist = [str(coord_list1D[c:c+3]) for c in range(0, len(coord_list1D), 3)]

np.savetxt("CoordNestList.csv", coord_nestlist, delimiter=",", fmt='%s')

#informing the user of the connection between the quantum numbers and the wavefunction probability plots
print('Each possible wavefunction for the hydrogen atom is defined by a set of three quantum numbers,'+
      '\nn the primary quantum number, which described the energy of the system,'+
      '\nl and m\u2097 the azumithal and magnetic quantum numbers,'+
      '\nwhich are related to the orbital angular momentum of the system, and describe the shape of each orbital.')

#asks the user to select a set of quantum number they want to inspect
n = int(input('Please input a value for n, from n=1 to n=4:'))
l = int(input('Please input a value for l, from l=0 to l=n-1:'))
m = int(input('Please input a value for m\u2097, from m\u2097=-l to m\u2097=l:'))

#checks to see if the combination of quantum numbers given is valid
if 0<n<=4 and 0<=l<n and -l<=m<=l:
    #-----------------Plotting the Probability Distribution ------------------#
    
    #sets the size of the plots
    fig = plt.figure(figsize=(10,6))
    #sets the axis for plot 1 to display in 3D, describes its position
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    #sets the position of plot 2
    ax2 = fig.add_subplot(1,2,2)

    #flattens the probability array for every point in space into a 1D array, converts to a list
    prob_list = np.ndarray.flatten(orbital_probability(n,l,m)[0],'F')

    #generates a list of random indices based on the our wavefunction probability
    rand_idx = np.random.choice(len(coord_nestlist), size=len(prob_list), p=prob_list)
    #uses the randomly selected indices to create an array that contains sets of coordinates
    coords = np.array(coord_nestlist)[rand_idx]
    #uses the same random indices to create an array of the probability associated with each set of coords
    probs = np.array(prob_list)[rand_idx]

    #splits the values coordinates generated with random.choice into a list of lists
    coord_list = [i.split(',') for i in coords]
    #loops through the nested list of coordinates, 
    #grabbing the x,y,z value of each nested list and converting it into a float
    x_coords = [float(i[0][1:]) for i in coord_list]
    y_coords = [float(i[1]) for i in coord_list]
    z_coords = [float(i[2][:-1]) for i in coord_list]

    #generates a 3D plot of our probability distribution
    prob_plot = ax1.scatter(x_coords, y_coords, z_coords, c = (probs*1000), alpha=0.5, s=(probs*1000))

    #plot the x,y and z axes to give the plot some dimension
    ax1.plot([(np.min(x_coords)),np.max(x_coords)], [0,0], [0,0], color = 'black', alpha=0.75)
    ax1.plot([0, 0], [(np.min(y_coords)),np.max(y_coords)], [0, 0], color = 'black', alpha=0.75)
    ax1.plot([0, 0], [0,0], [(np.min(z_coords)),np.max(z_coords)], color = 'black', alpha=0.75)

    #sets the plots title
    ax1.set_title('Probability Distribution\nfor an Electron in the n={0} l={1} m\u2097={2} State'.format(n, l, m))
    #setting axis labels
    ax1.set_xlabel('x-position (x/a\u2080)')
    ax1.set_ylabel('y-position (y/a\u2080)')
    ax1.set_zlabel('z-position (z/a\u2080)')
    #setting axis limits
    ax1.set_xlim((np.min(x_coords)*1.1),np.max(x_coords)*1.1)
    ax1.set_ylim((np.min(y_coords)*1.1),np.max(y_coords)*1.1)
    ax1.set_zlim((np.min(z_coords)*1.1),np.max(z_coords)*1.1)
   
    #sets a colorbar to illustrate the probability, resives it to better fit the plot space
    fig.colorbar(prob_plot, orientation="horizontal", ax = ax1, 
                 label = 'Probability Density |\u03A8(x,y,z)|\u00b2, Multiplied by a Factor of 1000', shrink = 0.75)

    #-----------------Plotting the Radial Component----------------#

    #flattens the array containing the r, converts to list
    radius_list = np.ndarray.flatten(r_array,'F').tolist()
    #flattens the radial component array for every point in space into a 1D array, converts to list
    radial_prob_list = np.ndarray.flatten(orbital_probability(n,l,m)[1],'F').tolist()

    #generates a 2D plot of the radial component of each wavefunction
    radial_plot = ax2.plot(radius_list,radial_prob_list, c = 'b')
    #sets the plots title
    ax2.set_title('Radial Probability Distribution\nfor an Electron in the n={0} l={1} State'.format(n, l))
    #setting axis labels
    ax2.set_xlabel('Distance from Nucleus (r/a\u2080)')
    ax2.set_ylabel('Radial Probability |\u03A8(r)|\u00b2')
    #sets maxs and mins for the x axis based on the value of n
    if n == 1:
        ax2.set_xlim(0,10)
    elif n == 2:
        ax2.set_xlim(0,15)
    elif n == 3:
        ax2.set_xlim(0,25)
    else:
        ax2.set_xlim(0,35)
    #sets maxs and mins for the y axis based on the max and min probabilities
    ax2.set_ylim(np.min(radial_prob_list)*1.1,np.max(radial_prob_list)*1.1)
    
    #sets gridlines, defines style
    ax2.grid(visible=True, which='major', color='#666666', linestyle='-')
    
    #saves the figures to a png file
    plt.savefig('n={0} l={1} m\u2097={2} 3D and Radial Probability Distribution.jpg'.format(n, l, m))
    
    #calculates the energy of the system
    E = -13.6/n**2
    
    #keeps a tight layout
    fig.tight_layout()
    
    #tells the user the energy of the system
    print('The binding or ionization energy for an electron in this state is {} eV'.format(E))

#if the combination of quantum numbers is invalid
else:
    #prints a statement informing the user of the problem
    print('This combination of quantum numbers is forbidden')
        
# Stop the timer 
t_stop = perf_counter()

print("Elapsed time during the whole program in seconds:", t_stop-t_start)

plt.show()