import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.special as special
from time import perf_counter
import math

# Start the timer
t_start = perf_counter()

# Define the Bohr radius
a_0 = 5.29e-11

# Probability function for the 1s orbital
def orbital_probability(n, l, m, r_array, phi_array, theta_array):
    '''This function computes a generalized probability density for the 1s orbital'''
    # Calculate the radial component of the wavefunction using scipy Laguerre polynomials
    radial_comp = (2 * r_array / n)**l * np.exp(-r_array / n) * special.genlaguerre(n - l - 1, 2 * l + 1)(2 * r_array / n)
    # Probability is the abs. square of a wavefunction
    radprob = np.abs(radial_comp)**2
    # Normalize the probability
    radprob /= radprob.sum()
    
    # Calculate the complete wavefunction using the radial component and scipy spherical harmonics
    wave_func = np.sqrt(((2 / (n * a_0))**3) * (math.factorial(n - l - 1) / math.factorial(2 * n * (n + l)))) * radial_comp * special.sph_harm(m, l, phi_array, theta_array)
    # Probability is the abs. of a wavefunction's complex conjugate times the wavefunction
    waveprob = np.abs(wave_func)**2
    # Normalize the probability
    waveprob /= waveprob.sum()
    # Return the full wavefunction probability distribution as well as its radial component
    return waveprob, radprob

# Define the x, y, z
y = np.linspace(-40, 40, 100)
z = np.linspace(-40, 40, 100)
x = np.linspace(-40, 40, 100)

# Generate 3D arrays for the x, y, z axis
x_array, y_array, z_array = np.meshgrid(x, y, z)

# Calculate the radius at each point in the grid
r_array = np.sqrt(x_array**2 + y_array**2 + z_array**2)
# Calculate the theta value for each point in the grid
phi_array = np.arctan2(y_array, x_array)
# Calculate the phi value for each point in the grid
theta_array = np.arctan2(np.sqrt(x_array**2 + y_array**2), z_array)

# Informing the user of the connection between the quantum numbers and the wavefunction probability plots
print('Each possible wavefunction for the hydrogen atom is defined by a set of three quantum numbers (integers),'
      '\nn the primary quantum number, which describes the energy of the system,'
      '\nl and m\u2097 the azimuthal and magnetic quantum numbers,'
      '\nwhich are related to the orbital angular momentum of the system, and describe the shape of each orbital.')

# Ask the user to select a set of quantum numbers they want to inspect
n = int(input('Please input a value for n, from n=1 to n=4:'))
l = int(input('Please input a value for l, from l=0 to n-1:'))
m = int(input('Please input a value for m\u2097, from m\u2097=-l to m\u2097=l:'))

# Check to see if the combination of quantum numbers given is valid
if 0 < n <= 4 and 0 <= l < n and -l <= m <= l:
    # Plotting the Probability Distribution
    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    # Get the probability arrays
    waveprob, radprob = orbital_probability(n, l, m, r_array, phi_array, theta_array)

    # Flatten the arrays for plotting
    x_coords = x_array.flatten()
    y_coords = y_array.flatten()
    z_coords = z_array.flatten()
    prob_list = waveprob.flatten()

    # Normalize probabilities for better visualization
    prob_list_normalized = prob_list / prob_list.max()
    # Hides probability values <0.05 for better clarity
    prob_list_normalized[prob_list_normalized<0.05]=0

    # Invert the normalized probabilities for opacity (less probable points are more opaque)
    opacity_list = (1-prob_list_normalized)*0.0075

    # Generate a 3D plot of our probability distribution
    prob_plot = ax1.scatter(x_coords, y_coords, z_coords, c=prob_list_normalized, cmap='viridis', alpha=opacity_list, s=(prob_list_normalized * 100))

    # Plot the x, y, and z axes to give the plot some dimension
    ax1.plot([np.min(x_coords), np.max(x_coords)], [0, 0], [0, 0], color='black', alpha=0.75)
    ax1.plot([0, 0], [np.min(y_coords), np.max(y_coords)], [0, 0], color='black', alpha=0.75)
    ax1.plot([0, 0], [0, 0], [np.min(z_coords), np.max(z_coords)], color='black', alpha=0.75)

    # Set the plot's title and labels
    ax1.set_title(f'Probability Distribution\nfor an Electron in the n={n} l={l} m\u2097={m} State')
    ax1.set_xlabel('x-position (x/a\u2080)')
    ax1.set_ylabel('y-position (y/a\u2080)')
    ax1.set_zlabel('z-position (z/a\u2080)')
    ax1.set_xlim(np.min(x_coords), np.max(x_coords))
    ax1.set_ylim(np.min(y_coords), np.max(y_coords))
    ax1.set_zlim(np.min(z_coords), np.max(z_coords))

    # Add a colorbar
    fig.colorbar(prob_plot, orientation="horizontal", ax=ax1, label='Normalized Probability Density |\u03A8(x,y,z)|\u00b2')

    # Plotting the Radial Component
    radius_list = r_array.flatten()
    radial_prob_list = radprob.flatten()

    # Generate a 2D plot of the radial component of each wavefunction
    ax2.plot(radius_list, radial_prob_list, c='b')
    ax2.set_title(f'Radial Probability Distribution\nfor an Electron in the n={n} l={l} State')
    ax2.set_xlabel('Distance from Nucleus (r/a\u2080)')
    ax2.set_ylabel('Radial Probability |\u03A8(r)|\u00b2')
    ax2.set_xlim(0, max(radius_list))
    ax2.set_ylim(np.min(radial_prob_list), np.max(radial_prob_list))
    ax2.grid(visible=True, which='major', color='#666666', linestyle='-')

    # Save the figures to a png file
    plt.savefig(f'n={n} l={l} m\u2097={m} 3D and Radial Probability Distribution.jpg')

    # Calculate the energy of the system
    E = -13.6 / n**2

    # Keep a tight layout
    fig.tight_layout()

    # Tell the user the energy of the system
    print(f'The binding or ionization energy for an electron in this state is {E} eV')

else:
    # Print a statement informing the user of the problem
    print('This combination of quantum numbers is forbidden')

# Stop the timer
t_stop = perf_counter()

print("Elapsed time during the whole program in seconds:", t_stop - t_start)

plt.show()
